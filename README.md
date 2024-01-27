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
```

### train

```bash
deepflame fit --config config.yaml \
    --trainer.max_epochs 100 # Append args if needed
```
