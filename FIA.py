# %% [markdown]
# # Fisher Information Analysis and Model Cross-Fitting
# 
# Code used to run analysis and generate plots for "Mitigating effects of jitter through differentiable
# forwards-modeling for the TOLIMAN space telescope".


# NOTE POSSIBLE LACK OF OVERSAMPLING CAUSING THE NUMBERICAL EFFECTS

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
from setup_jitter import setup_jitter


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

# %%
def fisher_sweep(
    tel,
    ll_fn,
    params,
    mags=np.linspace(1e-4, 2 * 0.375, 10),
    angs=np.linspace(0, 90, 3),
    save_memory=False,
):

    seps = []

    # looping over jitter magnitude
    for mag_idx, mag in tqdm(enumerate(mags), total=len(mags)):
        model = tel.set("jitter_mag", mag)

        # looping over jitter angle
        for ang_idx, ang in enumerate(angs):
            model = model.set("jitter_angle", ang)
            data = model.jitter_model()

            # calculate covariance matrix
            cov = zdx.covariance_matrix(
                tel, params, ll_fn, np.round(data), save_memory=save_memory
            )

            # read separation from covariance matrix
            sep = np.sqrt(np.abs(cov[0, 0]))
            seps.append(sep)

            # check for NaNs
            if np.isnan(sep).any():
                print(f"NaNs found at mag_idx={mag_idx}, ang_idx={ang_idx}")
                raise ValueError

    seps = np.array(seps).reshape(len(mags), len(angs))
    return seps

# %%
# Marginal parameters
params = [
    "separation",
    "position_angle",
    "x_position",
    "y_position",
    "log_flux",
    "contrast",
    "jitter_mag",
    "jitter_angle",
    "aperture.coefficients",
    # # 'wavelengths',
    # # 'psf_pixel_scale',
]

test_mags = np.linspace(1e-4, 2 * 0.375, 30)
test_angs = np.linspace(0, 90, 7)

# test_mags = np.linspace(1e-4, 2 * 0.375, 10)
# test_angs = np.linspace(0, 90, 3)

# %%
if run_compute:
    seps_lin = fisher_sweep(
        models["lin"],
        loglike_fns["lin"],
        params,
        mags=test_mags,
        angs=test_angs,
        save_memory=save_ram,
    )
    np.save(nt_files_path + "seps/seps_lin.npy", seps_lin)

# %%
if run_compute:
    seps_shm = fisher_sweep(
        models["shm"],
        loglike_fns["shm"],
        params,
        mags=test_mags,
        angs=test_angs,
        save_memory=save_ram,
    )
    np.save(nt_files_path + "seps/seps_shm.npy", seps_shm)

# %%
if run_compute:
    stable_model = models["lin"].set("jitter_mag", np.array(0.0))
    stable_data = stable_model.model()
    stable_cov = zdx.covariance_matrix(
        stable_model,
        params,
        loglike_fns["lin"],
        stable_data,
        save_memory=save_ram,
    )
    baseline = np.sqrt(np.abs(stable_cov[0, 0]))
    print(f"Baseline with no wavelengths or pixel scale: {1000 * baseline} mas")
    np.save(nt_files_path +"seps/baseline.npy", 1000 * baseline)

# %%
seps_lin = np.load(nt_files_path + "seps/seps_lin.npy")
seps_shm = np.load(nt_files_path + "seps/seps_shm.npy")
baseline = np.load(nt_files_path + "seps/baseline.npy")

# %%
det_pscale = 0.375
mags = test_mags
angs = test_angs
colors = ito_seven
# baseline = 0.24839681897145435  # mas

cmap = mpl.colormaps["cmr.bubblegum_r"]
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=90))
colors = cmap(angs / 90)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True, layout="compressed")

# looping over jitter type
for i, seps in enumerate(
    [
        seps_lin,
        seps_shm,
    ]
):

    # baseline
    if baseline is not None:
        ax[i].axhline(
            baseline, linestyle="--", c="k", alpha=0.4, label="No Jitter", linewidth=1
        )

    # looping over each line
    for sep, ang, c in zip(seps.T, angs, colors):
        ax[i].plot(
            mags,
            1000 * sep,
            label=r"$\phi\,=\,$" + f"{ang:.0f}",
            color=c,
            linewidth=2.5,
        )

    ax[i].tick_params(axis="x", which="both", top=False)

    # Add secondary x-axis scale showing units of pixels
    ax2 = ax[i].secondary_xaxis(
        "top", functions=(lambda x: x / det_pscale, lambda x: x * det_pscale)
    )
    ax2.set_xticks([0.0, 1.0, 2.0])  # Custom major xticks
    ax2.set_xticklabels([r"$0$", r"$1$", r"$2$"])  # Custom labels for major ticks
    ax2.minorticks_on()  # Enable minor ticks
    ax2.xaxis.set_minor_locator(
        mpl.ticker.AutoMinorLocator(5)
    )  # 5 minor ticks between major ticks

    ax2.set(xlabel="Jitter Excursion [pixels]", xticks=[0, 0.5, 1.0, 1.5, 2.0])
    ax[i].grid(True, alpha=0.2, linestyle="-", axis="y")

ax[0].set(
    xlabel="Jitter Excursion [arcsec]",
    ylabel=r"Separation Error $\sigma$ [mas]",
    xlim=(0, mags.max()),
)
ax[1].set(
    xlabel="Jitter Excursion [arcsec]",
    xlim=(0, mags.max()),
)

# adding text
for i, title in enumerate(["Linear Jitter", "SHM Jitter"]):
    ax[i].text(
        0.1,
        0.85,
        title,
        size=15,
        transform=ax[i].transAxes,
        ha="left",
        va="top",
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.3",
        ),
    )

cbar = fig.colorbar(sm, ax=ax, ticks=angs, label=r"$\phi$ (deg)")
cbar.ax.tick_params(direction="out")
cbar.ax.minorticks_off()

plt.savefig(nt_files_path + "paper_figs/lin_shm_sweep.png", bbox_inches="tight", dpi=500)
plt.savefig(nt_files_path + "paper_figs/lin_shm_sweep.pdf", bbox_inches="tight", dpi=500)
plt.close()
