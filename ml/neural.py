"""Small sklearn-compatible PyTorch scorer with portable CPU checkpoints."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


def resolve_device(requested='auto'):
    import torch
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable in this process. Run outside the sandbox or use --device cpu.')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable in this process')
    return requested


class TorchCandidateClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, hidden_size=64, epochs=15, batch_size=8192,
                 learning_rate=0.001, weight_decay=0.0001, seed=13, device='auto'):
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.seed = seed
        self.device = device

    def _network(self):
        import torch
        return torch.nn.Sequential(
            torch.nn.Linear(self.n_features_in_, self.hidden_size), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.hidden_size // 2), torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size // 2, 1))

    def fit(self, X, y, validation_data=None):
        import torch
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError('epochs and batch_size must be positive')
        self.device_ = resolve_device(self.device)
        torch.manual_seed(self.seed)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        if set(np.unique(y)) != {0, 1}:
            raise ValueError('Neural scorer requires both binary classes')
        net = self._network().to(self.device_)
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = torch.nn.BCEWithLogitsLoss()
        # Keep the feature matrix on-device; avoid per-batch host transfers.
        x = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device_)
        target = torch.as_tensor(np.asarray(y, dtype=np.float32), device=self.device_)
        rng = np.random.default_rng(self.seed)
        self.loss_curve_ = []
        self.validation_loss_curve_ = []
        if validation_data is not None:
            val_x = torch.as_tensor(np.asarray(validation_data[0], dtype=np.float32), device=self.device_)
            val_y = torch.as_tensor(np.asarray(validation_data[1], dtype=np.float32), device=self.device_)
        for _ in range(self.epochs):
            order = torch.as_tensor(rng.permutation(len(x)), device=self.device_)
            total = torch.zeros((), device=self.device_)
            net.train()
            for start in range(0, len(x), self.batch_size):
                idx = order[start:start + self.batch_size]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(net(x[idx]).squeeze(-1), target[idx])
                loss.backward()
                optimizer.step()
                total += loss.detach() * len(idx)
            self.loss_curve_.append(float(total.cpu()) / len(x))
            if validation_data is not None:
                net.eval()
                validation_loss = torch.zeros((), device=self.device_)
                with torch.inference_mode():
                    for start in range(0, len(val_x), self.batch_size):
                        end = start + self.batch_size
                        loss = criterion(net(val_x[start:end]).squeeze(-1), val_y[start:end])
                        validation_loss += loss * len(val_x[start:end])
                self.validation_loss_curve_.append(float(validation_loss.cpu()) / len(val_x))
        # numpy state avoids tying joblib bundles to Metal/CUDA at inference time.
        self.weights_ = {k: v.detach().cpu().numpy().copy() for k, v in net.state_dict().items()}
        return self

    def predict_proba(self, X):
        import torch
        check_is_fitted(self, 'weights_')
        # CPU inference is portable for the eventual backend and small batches.
        net = self._network()
        net.load_state_dict({k: torch.from_numpy(v) for k, v in self.weights_.items()})
        net.eval()
        result = []
        with torch.inference_mode():
            for start in range(0, len(X), self.batch_size):
                batch = torch.from_numpy(np.asarray(X[start:start+self.batch_size], dtype=np.float32))
                result.append(torch.sigmoid(net(batch).squeeze(-1)).numpy())
        prob = np.concatenate(result) if result else np.empty(0)
        return np.column_stack([1-prob, prob])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
