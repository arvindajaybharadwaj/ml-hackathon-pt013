"""Five model families and their training-only preprocessing."""
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

MODEL_NAMES = ['logistic_regression','gaussian_nb','decision_tree','random_forest','hist_gradient_boosting']

def build_model(name, seed=13, params=None, device='auto', epochs=15, threads=1):
    if name == 'torch_mlp':
        from .neural import TorchCandidateClassifier
        estimator = TorchCandidateClassifier(seed=seed, device=device, epochs=epochs)
        if params:
            estimator.set_params(**params)
        return Pipeline([('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
                         ('scale', StandardScaler()), ('model', estimator)])
    constructors = {
        'logistic_regression':lambda:LogisticRegression(max_iter=1500,C=1.,random_state=seed),
        'gaussian_nb':lambda:GaussianNB(var_smoothing=1e-9),
        'decision_tree':lambda:DecisionTreeClassifier(max_depth=8,min_samples_leaf=4,random_state=seed),
        'random_forest':lambda:RandomForestClassifier(n_estimators=80,max_depth=12,min_samples_leaf=2,n_jobs=threads,random_state=seed),
        # Internal row-wise early stopping is disabled: validation is by whole trip.
        'hist_gradient_boosting':lambda:HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=15,
                      learning_rate=.1,l2_regularization=1.,early_stopping=False,random_state=seed)}
    if name not in constructors:
        raise ValueError(f'Unknown model: {name}')
    estimator = constructors[name]()
    if params: estimator.set_params(**params)
    return Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
                     ('scale',StandardScaler()),('model',estimator)])


def suggest_parameters(trial, name):
    if name == 'torch_mlp':
        return {'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128]),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)}
    if name == 'logistic_regression': return {'C':trial.suggest_float('C',1e-3,100,log=True)}
    if name == 'gaussian_nb': return {'var_smoothing':trial.suggest_float('var_smoothing',1e-11,1e-3,log=True)}
    if name == 'decision_tree':
        return {'max_depth':trial.suggest_int('max_depth',3,18), 'min_samples_leaf':trial.suggest_int('min_samples_leaf',2,20)}
    if name == 'random_forest':
        return {'n_estimators':trial.suggest_int('n_estimators',40,140,step=20),
                'max_depth':trial.suggest_int('max_depth',4,18),
                'min_samples_leaf':trial.suggest_int('min_samples_leaf',1,12),
                'max_features':trial.suggest_float('max_features',.4,1.)}
    if name == 'hist_gradient_boosting':
        return {'max_iter':trial.suggest_int('max_iter',40,140,step=20),
                'max_leaf_nodes':trial.suggest_int('max_leaf_nodes',5,31),
                'learning_rate':trial.suggest_float('learning_rate',.02,.25,log=True),
                'l2_regularization':trial.suggest_float('l2_regularization',1e-3,10,log=True)}
    raise ValueError(name)
