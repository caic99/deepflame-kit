def boxcox(
    x,
    lmbda: float,
):  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox.html
    return (x**lmbda - 1) / lmbda  # if lmbda != 0 else log(x)


def inv_boxcox(y, lmbda: float):
    return (y * lmbda + 1) ** (1 / lmbda)  # if lmbda != 0 else exp(x)


epsilon = 1e-8


def normalize(x, mean, std):
    # TODO:handle the case for std = 0
    return (x - mean) / (std + epsilon)


def denormalize(y, mean, std):
    return y * (std + epsilon) + mean
