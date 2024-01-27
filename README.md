# DeepFlame-kit
DeepFlame-kit is a deep learning package for reacting flow.

## Usage
### Install

```bash
conda create -n pl lightning pytorch pytorch-cuda -c pytorch -c nvidia -c conda-forge
conda activate pl
pip install "jsonargparse[all]" wandb
```

### train

```bash
python ../../deepflame/main.py fit --config config.yaml --trainer.max_epochs 100
# Append args if needed
```
