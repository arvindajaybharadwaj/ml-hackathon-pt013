"""Full real-data training on Modal. Launch after the local benchmark."""
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App('pathsnap-training')
volume = modal.Volume.from_name('pathsnap-training-results', create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install_from_requirements(str(ROOT / 'ml' / 'requirements.txt'))
    .pip_install('torch==2.12.1')
    .add_local_python_source('ml')
    .add_local_dir(ROOT / 'prepared_data', '/input/prepared_data')
    .add_local_file(ROOT / 'feature_schema.json', '/input/feature_schema.json')
)


@app.function(image=image, cpu=8, memory=16384, gpu='T4', timeout=14400,
              volumes={'/results': volume})
def train(run_id: str, trials: int, epochs: int):
    import subprocess
    import sys
    import tarfile
    output = Path('/results') / run_id
    # All five CPU baselines plus a CUDA neural scorer; no raw reference files shipped.
    command = [sys.executable, '-m', 'ml', '--input', '/input/prepared_data',
               '--output', str(output), '--folds', '3', '--trials', str(trials),
               '--threads', '8', '--models', 'logistic_regression', 'gaussian_nb',
               'decision_tree', 'random_forest', 'hist_gradient_boosting', 'torch_mlp',
               '--device', 'cuda', '--epochs', str(epochs)]
    try:
        subprocess.run(command, check=True)
        archive = Path('/results') / f'{run_id}.tar.gz'
        with tarfile.open(archive, 'w:gz') as bundle:
            bundle.add(output, arcname=run_id)
        return archive.name
    finally:
        # Preserve partial results as well if training fails.
        volume.commit()


@app.local_entrypoint()
def main(trials: int = 20, epochs: int = 15, run_id: str = ''):
    from datetime import datetime, timezone
    if not run_id:
        run_id = 'modal-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('run_id must contain letters, digits, underscore or dash')
    if trials != 0 and trials < 4:
        raise ValueError('trials must be zero or at least four')
    archive = train.remote(run_id, trials, epochs)
    destination = ROOT / 'runs' / archive
    destination.parent.mkdir(exist_ok=True)
    with destination.open('wb') as stream:
        for chunk in volume.read_file(archive):
            stream.write(chunk)
    # Only extract the archive produced by this job, with safe relative members.
    import tarfile
    with tarfile.open(destination) as bundle:
        bundle.extractall(ROOT / 'runs', filter='data')
    print(f'Results downloaded to {ROOT / "runs" / run_id}')
