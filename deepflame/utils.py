def boxcox(
    x, lmbda=0.05
):  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.boxcox.html
    return (x**lmbda - 1) / lmbda  # if lmbda != 0 else log(x)


def inv_boxcox(y, lmbda=0.05):
    return (y * lmbda + 1) ** (1 / lmbda)  # if lmbda != 0 else exp(x)
