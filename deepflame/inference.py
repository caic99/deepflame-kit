import torch
import yaml
from deepflame.utils import inv_boxcox


def parse_properties(config_path: str) -> dict:
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
    property_config_path="/root/DeepFlame-examples/1Dflame/Tu800K-Phi1.0/constant/CanteraTorchProperties",
    model_config_path="/root/deepflame-kit/examples/hy41/config.yaml",
    checkpoint_path="/root/deepflame-kit/examples/hy41/dfnn/bbfus18q/checkpoints/epoch=2-step=207.ckpt",
) -> torch.nn.Module:
    from lightning.pytorch.cli import LightningCLI
    from deepflame.data import DFDataModule
    from deepflame.model import DFNN

    property_config = parse_properties(property_config_path)
    settings = property_config["TorchSettings"]
    assert (
        settings["torch"] == True
    ), f"torch is not set to 'on' in {property_config_path}"
    if settings["GPU"] == True:
        assert (
            torch.cuda.is_available()
        ), f"GPU is set to 'on' but not available in {property_config_path}"
    if model_config_path is None:
        model_config_path = settings["torchModel"]

    cli_args = f"predict --config {model_config_path} --trainer.limit_predict_batches=0".split()
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


# Inference API: https://github.com/deepmodeling/deepflame-dev/blob/master/src/dfChemistryModel/pytorchFunctions.H
# Currently `inference()` is called directly from C++,
# so we have to explicitly put the model in the scope of this file.
# TODO: fix this
model: torch.nn.Module = load_lightning_model()
n_species = model.state_dict()["model.formation_enthalpies"].shape[0]
time_step = model.state_dict()["model.time_step"]


# @torch.compile()
def inference(input):
    # TODO: check the input type: is it a Tensor or numpy array?
    # input shape: batch * [T, P, Y_ns, rho]
    input = input.to(model.device)
    # input = input[input[:, 0] >= settings["frozenTemperature"]] # FIXME: for debug only

    T, P, Y_in, rho = torch.split(input, [1, 1, n_species, 1], dim=1)

    with torch.no_grad():
        Y_t = model.forward(T, P, Y_in)
    Y = inv_boxcox(Y_t)
    # Why model has no attribute time_step?
    # return the mass change rate
    return (Y - Y_in) * (rho / time_step)
