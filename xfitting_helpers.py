from jax import numpy as np
from dLuxToliman import Toliman, JitteredToliman
import matplotlib as mpl
from matplotlib import colors, colormaps, pyplot as plt
import dLuxToliman as dlT

from zodiax.base import Base, _get_leaves, _format
from typing import Union, List
from jaxtyping import Array
from jax.flatten_util import ravel_pytree
from equinox import tree_at


Params = Union[str, List[str]]


def fwhm_to_det(fwhm, shear):
    return 1e6 * (1 - shear) ** 2 * (fwhm / 2.35482) ** 4


def det_to_fwhm(det, shear):
    return 2.35482 * ((det / 1e6) / (1 - shear) ** 2) ** 0.25


def get_shear(fwhm, det):
    return 1 - np.sqrt((det / 1e6) / (fwhm / 2.35482) ** 4)


def powspace(start, stop, power, num):
    """
    To generate r values at appropriate intervals.
    """
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


class NGDToliman(Toliman):
    def matmul(self: Base, parameters: Params, matrix: Array) -> Base:
        """
        Left matmul method.
        """
        new_parameters = _format(parameters)
        leaves = _get_leaves(self, new_parameters)

        # dealing with array leaves with size > 1
        values, unravel_fn = ravel_pytree(leaves)
        new_values = matrix @ values  # matrix multiplication
        new_leaves = unravel_fn(new_values)

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return tree_at(leaves_fn, self, new_leaves, is_leaf=lambda leaf: leaf is None)


class NGDJitteredToliman(JitteredToliman):
    def matmul(self: Base, parameters: Params, matrix: Array) -> Base:
        """
        Left matmul method.
        """
        new_parameters = _format(parameters)
        leaves = _get_leaves(self, new_parameters)

        # dealing with array leaves with size > 1
        values, unravel_fn = ravel_pytree(leaves)
        new_values = matrix @ values  # matrix multiplication
        new_leaves = unravel_fn(new_values)

        # Define 'where' function and update pytree
        def leaves_fn(pytree):
            return _get_leaves(pytree, new_parameters)

        return tree_at(leaves_fn, self, new_leaves, is_leaf=lambda leaf: leaf is None)


def plot_losses(
    losses,
    save_path,
    start,
    stop=-1,
    suffix="",
):
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.title("Full Loss")
    plt.plot(losses)

    if start >= len(losses):
        start = 0
    last_losses = losses[start:stop]
    n = len(last_losses)
    plt.subplot(1, 2, 2)
    plt.title(f"Final {n} Losses")
    plt.plot(np.arange(start, start + n), last_losses)

    plt.tight_layout()
    plt.savefig(save_path + f"test/losses{suffix}.png", dpi=150)
    plt.close()


def summarise_fit(
    model,
    data,
    loglike_fn,
    save_path,
    suffix="",
):

    inferno = colormaps["inferno"]
    seismic = colormaps["seismic"]
    inferno.set_bad("k", 0.5)
    seismic.set_bad("k", 0.5)

    if not isinstance(model, dlT.JitteredToliman):
        sim = model.model()
    elif isinstance(model, dlT.JitteredToliman):
        sim = model.jitter_model()
    residual = data - sim

    loglike_im = loglike_fn(model, data)
    final_loss = np.nanmean(-loglike_im)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"Pixel neg log posterior: {final_loss:,.1f}")
    plt.imshow(-loglike_im, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Mean noise mvnalised slope residual")
    plt.imshow(residual, cmap=seismic, norm=colors.CenteredNorm())
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(save_path + f"test/map{suffix}.png", dpi=150)
    plt.close()

    if not isinstance(model, dlT.JitteredToliman):
        try:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.title(f"Simulation")
            plt.imshow(sim, cmap="inferno")
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.title("Convolution Kernel")
            plt.imshow(
                model.Jitter.generate_kernel(1.00),
                cmap="cividis",
                norm=mpl.colors.LogNorm(),
            )
            plt.colorbar()

            plt.tight_layout()
            plt.savefig(save_path + f"test/kernel{suffix}.png", dpi=150)
            plt.close()
        except:
            pass
