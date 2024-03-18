"""
The inference API for DeepFlame.
Usage:
    from deepflame.inference import inference
    input = torch.stack(T, P, Y_ns, rho, axis=1)
    output = inference(input)
"""


import numpy as np
import torch
import yaml


def parse_properties(config_path) -> dict:
    with open(config_path, "r") as f:
        data = f.read()
        # remove comments marked with /* and */
        while i := data.find("/*") != -1:
            j = data.find("*/", i)
            assert j != -1, f"unmatched /* in {config_path}"
            data = data[: i - 1] + data[j + 3 :]
        bra = "{"
        ket = "}"
        indent = 0
        lines = data.split("\n")
        for i, line in enumerate(lines):
            # remove comments marked with //
            if (p := line.find("//")) != -1:
                line = line[:p]
            # remove trailing ';'
            line = line.strip().strip(";")
            # add indentation for lines between { and }
            if bra in line:
                line = ""
                indent += 4
            if ket in line:
                line = ""
                indent -= 4
            assert indent >= 0, f"unmatched {bra} in {config_path}"
            token = line.split()
            if len(token) == 0:
                pass
            elif len(token) == 1:
                line += ":"
            elif len(token) == 2:
                line = token[0] + ": " + token[1]
            else:
                assert False, f"invalid line: {line}"
            line = " " * indent + line
            lines[i] = line
        assert indent == 0, f"unmatched {ket} in {config_path}"

        # convert the content to yaml
        data = "\n".join(lines)
        # "on" and "off" are converted to True and False
        # https://yaml.org/type/bool.html
        properties = yaml.load(data, Loader=yaml.Loader)
        return properties


def load_lightning_model(
    model_config_path,
    checkpoint_path,
) -> torch.nn.Module:
    from lightning.pytorch.cli import LightningCLI
    from deepflame.data import DFDataModule
    from deepflame.model import DFNN

    cli_args = f"predict --config {model_config_path} --ckpt_path {checkpoint_path} --trainer.limit_predict_batches=0 --trainer.logger=False".split()
    # Generate model from config file
    cli = LightningCLI(
        DFNN,
        DFDataModule,
        args=cli_args,  # type: ignore
        # https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html#run-from-python
        save_config_callback=None,
    )
    # https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_3.html#instantiation-only-mode
    # Unfortunately, run=False does not work for it will not configure argument`ckpt`
    # So here, we use `predict` and set `limit_predict_batches=0` to avoid running actually.
    return cli.model


def load_torch_model(
    model_config_path,
    checkpoint_path,
) -> torch.nn.Module:
    from deepflame.model import DFNN
    config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    module = DFNN(**config["model"])
    state_dict = torch.load(checkpoint_path)["state_dict"]
    missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
    assert len(missing_keys) == 0, f"{missing_keys=}"
    for k in unexpected_keys:
        v = state_dict[k]
        module.model.register_buffer(
            k.removeprefix("model."),
            (
                v.detach().clone().type(torch.get_default_dtype())
                if isinstance(v, torch.Tensor)
                else torch.tensor(v)
            ),
        )
    module.load_state_dict(state_dict, strict=True)
    return module


property_config_path = (
    "/root/DeepFlame-examples/1Dflame/Tu800K-Phi1.0/constant/CanteraTorchProperties"
)

property_config = parse_properties(property_config_path)
settings = property_config["TorchSettings"]
frozen_temperature = settings["frozenTemperature"]
checkpoint_path = settings["torchModel"]
# Inference API: https://github.com/deepmodeling/deepflame-dev/blob/master/src/dfChemistryModel/pytorchFunctions.H
# Currently `inference()` is called directly from C++,
# so we have to explicitly put the model in the scope of this file.

model_config_path="/root/deepflame-kit/examples/hy41/config.yaml"
# checkpoint_path = "/root/deepflame-kit/examples/hy41/dfnn/rsw0yvsf/checkpoints/epoch=145-step=40004.ckpt"
checkpoint_path = "/root/deepflame-kit/examples/hy41/dfnn/lnn7u6iu/checkpoints/epoch=7-step=2192.ckpt"
# TODO: extract from config file

assert settings["torch"] == True, f"torch is not set to 'on' in {property_config_path}"
if settings["GPU"] == True:
    assert (
        torch.cuda.is_available()
    ), f"GPU is set to 'on' in {property_config_path}, but no GPU is available."

# Check inert index matches
print("Model config path: ", model_config_path)
model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
mechanism_file = property_config["CanteraMechanismFile"]
print("Cantera mechanism file: ", mechanism_file)
mechanism = yaml.load(open(mechanism_file, "r"), Loader=yaml.FullLoader)
species = mechanism["phases"][0]["species"] # ["name"] == "gas"
inert_specie = property_config["inertSpecie"]
print("Inert specie: ", inert_specie)
assert species[model_config["data"]["inert_index"]] == inert_specie, f"Inert specie does not match"

default_device = "cuda:0"
# torch.set_default_device(default_device)
load_model=load_torch_model
# load_model = load_lightning_model
print("Checkpoint path: ", checkpoint_path)
module: torch.nn.Module = load_model(model_config_path, checkpoint_path)
module = module.to(default_device).eval()
n_species: int = module.model.formation_enthalpies.shape[0]
time_step: float = module.model.time_step
lmbda: float = module.model.lmbda


# @torch.compile()
def inference(input_array: np.ndarray) -> np.ndarray:
    # input shape: batch * [T, P, Y_ns, rho] (flattened to 1D)
    input = torch.Tensor(input_array).to(default_device).reshape(-1, 1 + 1 + n_species + 1)  # .abs()
    mask = input[:, 0] >= frozen_temperature
    input_selected = input[mask]
    T, P, Y_in, rho = torch.split(input_selected, [1, 1, n_species, 1], dim=1)
    P[:,:]=101325. # Otherwise the model would diverge. This may be related to the sampling strategy of the training data.
    with torch.no_grad():
        Y_delta = module.predict(T, P, Y_in)
    # return the mass change rate
    rate = Y_delta * (rho / time_step)
    Y_out = torch.zeros([input.shape[0], n_species]).to(default_device)
    Y_out[mask] = rate
    return Y_out.cpu().numpy()
