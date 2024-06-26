# DeepFlame-kit
DeepFlame-kit is a deep learning package for reacting flow.

## Usage
### Install

```bash
# First activate your deepflame environment
git clone https://github.com/caic99/deepflame-kit.git
cd deepflame-kit
pip install -e . # Will also install requirements
deepflame --help
```

### Training

```bash
# First configure setup in `config.yaml`
cd deepflame-kit/examples/hy41
deepflame fit --config config.yaml
#    --trainer.max_epochs 100 --model.dec_size=[512,128] # Append args if needed
# deepflame fit --help
```

The checkpoint with the best validation loss will be saved as a `.ckpt` file.

### Infer

```bash
# Dataset Reference: https://nb.bohrium.dp.tech/competitions/detail/8918899584?tab=datasets
cd deepflame-kit/examples/Tu800K-Phi1.0
# !! Set the model config and weight paths in `constant/CanteraTorchProperties`
source /opt/OpenFOAM-7/etc/bashrc
source ~/deepflame-dev/bashrc
rm -r 0.00*
./Allclean
./Allrun
reconstructPar
flameSpeed
# Reference value: flamePoint.x = 0.01011, flameSpeed = 2.14 m/s
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
