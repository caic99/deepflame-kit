# DeepFlame-kit
DeepFlame-kit is a deep learning package for reacting flow.

## Usage
### Install

```bash
# Create a new virtual environment
# conda create -n pl lightning pytorch pytorch-cuda -c pytorch -c nvidia -c conda-forge
# conda activate pl

git clone https://github.com/caic99/deepflame-kit.git
cd deepflame-kit
pip install . -e
pip install "jsonargparse[all]" wandb # optional
deepflame --help
```

### Training

```bash
deepflame fit --config config.yaml \
    --trainer.max_epochs 100 # Append args if needed
deepflame fit --help
```

## Development
### File Structure

```yaml
deepflame: package and utils
- data.py: dataset and dataloader setup
- model.py: model definition
- trainer.py: training loop
- main.py: entrypoint
- utils.py: helper functions

examples:
- hy41: aviation kerosene, 41 species
  - config.yaml: training config
  - dataset.npy: training data and label (omitted for large size)
  - formation_enthalpies.npy: formation enthalpies of species
```
