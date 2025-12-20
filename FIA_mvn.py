# %% [markdown]
# # Fisher Information Analysis and Model Cross-Fitting
# 
# Code used to run analysis and generate plots for "Mitigating effects of jitter through differentiable
# forwards-modeling for the TOLIMAN space telescope".

# %%
import os

import jax
from jax import numpy as np, random as jr, Array
import zodiax as zdx
import dLux as dl
import dLuxToliman as dlT

# Enable 64bit precision (note this must be run in the first cell of the notebook)
# jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

import optax
from tqdm import tqdm
import random
from datetime import datetime

# plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
import scienceplots
import cmasher as cmr

plt.style.use(["science", "no-latex"])
plt.rcParams["image.origin"] = "lower"
plt.rcParams["figure.dpi"] = 300

# Colour schemes
ito_seven = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
    "#F0E442",
]
contrast_three = ["#004488", "#BB5566", "#DDAA33"]

# %%
# Set to True if you want to run the computation
# Set to False if you want to load the pre-calculated results from disk
run_compute = True
save_ram = False
nt_files_path = "/fred/oz440/max/code/tolicode/files/"

# %% [markdown]
# Setting up the model.

# %% [markdown]
# ## Priors

# %%
from jax.scipy.stats import norm, beta


def weibull_logpdf(x, k=1.5, lam=35):
    x = np.asarray(x)
    return np.log(k) - k * np.log(lam) + (k - 1) * np.log(x) - (x / lam) ** k



# %%
from setup_jitter import setup_jitter, powspace, fwhm_to_det, det_to_fwhm


def prior_fn(model, args={}):

    prior = 0

    if isinstance(model, dlT.JitteredToliman):
        # jitter angle phi
        angle = model.get("jitter_angle")
    elif isinstance(model, dlT.Toliman):
        # determinant r
        det = model.get("Jitter.r")
        prior += weibull_logpdf(det)

        # shear
        shear = model.get("Jitter.shear")
        prior += beta.logpdf(shear, a=1.1, b=1.1)

        # jitter angle phi
        angle = model.get("Jitter.phi")
    prior += norm.logpdf(x=angle, loc=args["angle"], scale=1.0)

    # aberration coefficients Z
    aberrations = model.get("aperture.coefficients")
    prior += norm.logpdf(x=aberrations, loc=0.0, scale=4.0).sum()

    return prior


models, datas, params, loglike_fns, posterior_fns = setup_jitter(
    oversample=4,
    n_psfs=5,
    prior_fn=prior_fn,
)

# %% [markdown]
# ## Linear & SHM Models: FIA
# 
# The Fisher Information analysis for the Linear and SHM jitter models, and their respective plots.


# Marginal params for normal model
norm_params = [
    "separation",
    "position_angle",
    "x_position",
    "y_position",
    "log_flux",
    "contrast",
    "Jitter.r",
    "Jitter.shear",
    "Jitter.phi",
    "aperture.coefficients",
    # 'wavelengths',
    # 'psf_pixel_scale',
]

det_pscale = models["norm"].psf_pixel_scale
oversample = models["norm"].oversample
kernel_size = models["norm"].Jitter.kernel_size

phis = np.linspace(0, 90, 7)
shears = np.array([0, 0.3, 0.7])
rs = powspace(1e-1, fwhm_to_det(1.01 * det_pscale, shears[0]), 2, 5)
# rs = powspace(1e-7, fwhm_to_det(1.01 * det_pscale, shears[0]), 2, 30)

if run_compute:

    seps = []
    fwhms = []
    kernels = []

    for shear_idx, shear in enumerate(shears):
        model = models["norm"].set("detector.Jitter.shear", shear)

        for r_idx, r in tqdm(enumerate(rs), total=len(rs)):
            fwhm = det_to_fwhm(r, shear)
            fwhms.append(fwhm)
            model = model.set("detector.Jitter.r", r)

            for phi_idx, phi in enumerate(phis):

                # skipping over different angles for shear = 0 as symmetric
                if shear_idx == 0 and phi_idx != 0:
                    sep = np.nan

                else:
                    model = model.set("detector.Jitter.phi", phi)
                    data = model.model()

                    # cov = cov_fns["norm"](model, np.round(data), norm_params)
                    cov = zdx.covariance_matrix(
                        model, norm_params, loglike_fns["norm"], data, save_memory=save_ram
                    )
                    sep = np.sqrt(np.abs(cov[0, 0]))
                    if phi_idx == 0:
                        if r == rs.max():
                            kernels.append(
                                model.Jitter.generate_kernel(det_pscale / oversample)
                            )

                seps.append(sep)

    seps = np.array(seps).reshape(len(shears), len(rs), len(phis))
    fwhms = np.array(fwhms).reshape(len(shears), len(rs))

    # saving
    np.save(nt_files_path + "seps/seps_norm.npy", seps)
    np.save(nt_files_path + "seps/kernels.npy", kernels)
    np.save(nt_files_path + "seps/fwhms.npy", fwhms)


if run_compute:
    stable_model = models["norm"].set("Jitter.r", np.array(1e-8))
    stable_data = stable_model.model()
    stable_cov = zdx.covariance_matrix(
        stable_model,
        norm_params,
        loglike_fns["norm"],
        np.round(stable_data),
        save_memory=save_ram,
    )
    baseline = np.sqrt(np.abs(stable_cov[0, 0]))
    print(f"Baseline with no wavelengths or pixel scale: {1000 * baseline} mas")
    np.save(nt_files_path +"seps/baseline_mvn.npy", 1000 * baseline)

seps_norm = np.load(nt_files_path + "seps/seps_norm.npy")
kernels = np.load(nt_files_path + "seps/kernels.npy")
fwhms = np.load(nt_files_path + "seps/fwhms.npy")
baseline = np.load(nt_files_path + "seps/baseline_mvn.npy")


cmap = mpl.colormaps["cmr.gem"]
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=90))
colors = cmap(phis / 90)

fig, axes = plt.subplots(
    len(shears), 2, figsize=(9, 7), sharey="col", layout="compressed"
)

for i, shear in enumerate(shears):
    ax, axe = axes[i]
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 2))
    ax.tick_params(axis="x", which="both", top=False)

    for sep, phi, c in zip(seps_norm[i].T, phis, colors):
        if shear == 0:
            label = None
            c = "k"
        else:
            label = label = r"$\phi\,=$" + f" {phi:.0f}"
        ax.plot(rs, 1000 * sep, label=label, color=c, marker=None, linewidth=2.75)
    ax.set(
        xlabel=r"det$\,\Sigma$ [arcsec$^{4}$]",
        ylabel=r"Separation Error $\sigma$ [mas]",
        # ylim=(baseline - 0.01, 0.8),
        xlim=(0, rs.max()),
    )

    ax.text(
        0.27,
        0.87,
        rf"$\eta={shear}$",
        size=12,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle=None),
    )
    ax.axhline(
        baseline, linestyle="--", c="k", alpha=0.4, label="No Jitter", linewidth=1
    )

    if shear == 0:
        cbar = fig.colorbar(sm, ax=ax)
        cbar.ax.set_visible(False)  # Hide the colorbar visually
    else:
        cbar = fig.colorbar(sm, ax=ax, ticks=phis, label=r"$\phi$ [deg]")
        cbar.ax.minorticks_off()  # Ensure minor ticks are disabled
        cbar.ax.tick_params(direction="out")

    extent = (
        np.array([-kernel_size / 2, kernel_size / 2, -kernel_size / 2, kernel_size / 2])
        * det_pscale
        / oversample
    )
    c = axe.imshow(
        kernels[i],
        cmap="cmr.arctic_r",
        origin="lower",
        extent=extent,
    )
    axe.set(
        xticks=[],
        yticks=[],
    )
    axe.minorticks_off()

    max_fwhm = det_to_fwhm(rs.max(), shear)
    axe.hlines(
        0,
        -max_fwhm / 2,
        max_fwhm / 2,
        colors="violet",
        linestyles="solid",
        label="FWHM",
        linewidth=2.5,
    )

    for spine in ["top", "right", "left", "bottom"]:
        axe.spines[spine].set_visible(False)

    if shear == 0:
        axe.set_title("Convolution Kernel")
        axe.text(
            0.0,
            0.05,
            "FWHM",
            size=8,
            ha="center",
            va="center",
            c="violet",
            weight="bold",
            fontname="Georgia",
        )


# SECONDARY AXES (was buggy so had to remove from loop)

# First subplot
ax00 = axes[0][0].secondary_xaxis(
    "top",
    functions=(
        lambda r: det_to_fwhm(r, shears[0]),
        lambda fwhm: fwhm_to_det(fwhm, shears[0]),
    ),
)
ax00.set_xticks([0.2, 0.3, 0.35])  # Custom major xticks
ax00.set_xticklabels([r"$0.2$", r"$0.3$", r"$0.35$"])  # Custom labels for major ticks
ax00.minorticks_on()  # Enable minor ticks
ax00.xaxis.set_minor_locator(
    mpl.ticker.AutoMinorLocator(5)
)  # 5 minor ticks between major ticks
ax00.set_xlabel("FWHM of semi-major axis [arcsec]")

# Second subplot
ax11 = axes[1][0].secondary_xaxis(
    "top",
    functions=(
        lambda r: det_to_fwhm(r, shears[1]),
        lambda fwhm: fwhm_to_det(fwhm, shears[1]),
    ),
)
ax11.set_xticks([0.2, 0.3, 0.4, 0.45])  # Custom major xticks
ax11.set_xticklabels(
    [r"$0.2$", r"$0.3$", r"$0.4$", r"$0.45$"]
)  # Custom labels for major ticks
ax11.minorticks_on()  # Enable minor ticks
ax11.xaxis.set_minor_locator(
    mpl.ticker.AutoMinorLocator(5)
)  # 5 minor ticks between major ticks
ax11.set_xlabel("FWHM of semi-major axis [arcsec]")

# First subplot
ax22 = axes[2][0].secondary_xaxis(
    "top",
    functions=(
        lambda r: det_to_fwhm(r, shears[2]),
        lambda fwhm: fwhm_to_det(fwhm, shears[2]),
    ),
)
ax22.set_xticks([0.4, 0.5, 0.6, 0.65])  # Custom major xticks
ax22.set_xticklabels(
    [r"$0.4$", r"$0.5$", r"$0.6$", r"$0.65$"]
)  # Custom labels for major ticks
ax22.minorticks_on()  # Enable minor ticks
ax22.xaxis.set_minor_locator(
    mpl.ticker.AutoMinorLocator(5)
)  # 5 minor ticks between major ticks
ax22.set_xlabel("FWHM of semi-major axis [arcsec]")

plt.savefig(nt_files_path + "paper_figs/norm_sweep.pdf", bbox_inches="tight", dpi=500)
plt.savefig(nt_files_path + "paper_figs/norm_sweep.png", bbox_inches="tight", dpi=500)
plt.close()