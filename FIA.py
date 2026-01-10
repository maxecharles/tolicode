# %% [markdown]
# # Fisher Information Analysis and Model Cross-Fitting
# 
# Code used to run analysis and generate plots for "Mitigating effects of jitter through differentiable
# forwards-modeling for the TOLIMAN space telescope".


# NOTE POSSIBLE LACK OF OVERSAMPLING CAUSING THE NUMBERICAL EFFECTS

# %%
import jax
# Enable 64bit precision (note this must be run in the first cell of the notebook)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

print(jax.devices())
print(jax.config.jax_enable_x64)

import os
from jax import numpy as np, random as jr, Array
import zodiax as zdx
import dLux as dl
import dLuxToliman as dlT


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

import dLuxToliman as dlT
import dLux
from dLuxToliman import AlphaCen
import jax
from jax import numpy as np, Array
import jax.scipy as jsp

import zodiax as zdx
from zodiax import filter_vmap
from setup_jitter import fwhm_to_det, det_to_fwhm, powspace

# from setup_jitter import setup_jitter
def setup_jitter(
    angle=0.0,
    mag=0.5 * 0.375,
    # shear=0.1,
    # r=fwhm_to_det(0.5 * 0.375, 0.1),
    oversample=4,
    # norm_osamp=6,
    det_pscale=0.375,
    det_npixels=128,
    # kernel_size=17,
    n_psfs=5,
    prior_fn=lambda model: np.array(0.0),
):
    lin_params = {
        "jitter_mag": mag,
        "jitter_angle": angle,
        "jitter_shape": "linear",
        "n_psfs": n_psfs,
    }
    shm_params = {
        "jitter_mag": mag,
        "jitter_angle": angle,
        "jitter_shape": "shm",
        "n_psfs": n_psfs,
    }
    # norm_params = {"r": r, "shear": shear, "phi": angle, "kernel_size": kernel_size}
    radial_orders = [2, 3]

    # Creating common optical system
    print("Building the models...")
    optics = dlT.TolimanOpticalSystem(
        oversample=oversample,
        psf_npixels=det_npixels,
        radial_orders=radial_orders,
        psf_pixel_scale=det_pscale,
    )
    optics = optics.divide("aperture.basis", 1e9)  # Set basis units to nanometers
    # norm_optics = optics.set("oversample", norm_osamp)

    # Creating common source
    src = AlphaCen(
        separation=np.array(10.0),
        position_angle=np.array(90.0),
        x_position=np.array(0.0),
        y_position=np.array(0.0),
        log_flux=np.array(7.581),
        contrast=np.array(3.37),
    )

    # creating telescopes
    lin_det = dLux.LayeredDetector([("Downsample", dLux.Downsample(oversample))])
    shm_det = lin_det
    # norm_det = dLux.LayeredDetector(
    #     [
    #         ("Jitter", dlT.GaussianJitter(**norm_params)),
    #         ("Downsample", dLux.Downsample(norm_osamp)),
    #     ]
    # )

    # creating models
    lin_tel = dlT.JitteredToliman(source=src, optics=optics, **lin_params).set(
        "detector", lin_det
    )
    shm_tel = dlT.JitteredToliman(source=src, optics=optics, **shm_params).set(
        "detector", shm_det
    )
    # norm_tel = dlT.Toliman(source=src, optics=norm_optics).set("detector", norm_det)

    # creating simulated data at a high oversample
    print("Creating simulated data grid...")
    # dlin_tel = lin_tel.set(["oversample", "Downsample.kernel_size"], [8, 8])
    dlin_tel = lin_tel
    lin_datas = []

    # dshm_tel = shm_tel.set(["oversample", "Downsample.kernel_size"], [8, 8])
    dshm_tel = shm_tel
    shm_datas = []

    # dnorm_tel = norm_tel.set(["oversample", "Downsample.kernel_size"], [8, 8])
    # dnorm_tel = norm_tel
    # norm_datas = []

    for ang in np.linspace(0, 90, 5):
        for mag in np.linspace(0.375 / 5, 0.375 / 1, 5):
            dlin_tel = dlin_tel.set("jitter_mag", mag).set("jitter_angle", ang)
            lin_data = {
                "params": ["jitter_mag", "jitter_angle"],
                "values": [mag, ang],
                "data": dlin_tel.jitter_model(),
            }
            lin_datas.append(lin_data)

            dshm_tel = dshm_tel.set("jitter_mag", mag).set("jitter_angle", ang)
            shm_data = {
                "params": ["jitter_mag", "jitter_angle"],
                "values": [mag, ang],
                "data": dshm_tel.jitter_model(),
            }
            shm_datas.append(shm_data)

        # for r in np.linspace(
        #     fwhm_to_det(0.375 / 5, 0.1), fwhm_to_det(0.375 / 1, 0.1), 5
        # ):
        #     dnorm_tel = dnorm_tel.set("Jitter.r", r).set("Jitter.phi", ang)
        #     norm_data = {
        #         "params": ["Jitter.r", "Jitter.phi"],
        #         "values": [r, ang],
        #         "data": dnorm_tel.model(),
        #     }
        #     norm_datas.append(norm_data)


    # NOTE make sure to ROUND data before passing here.
    likelihood_fn = lambda model, data: jsp.stats.poisson.logpmf(
        data, model.jitter_model()
    ).sum()

    posterior_fn = lambda model, data, args: likelihood_fn(model, data) + prior_fn(
        model, args
    )

    # norm_likelihood_fn = lambda model, data: jsp.stats.poisson.logpmf(
    #     data, model.model()
    # ).sum()

    # norm_posterior_fn = lambda model, data, args: norm_likelihood_fn(
    #     model, data
    # ) + prior_fn(model, args)

    # Wrapping everything up and returning
    models = {"lin": lin_tel, "shm": shm_tel, 
        # "norm": norm_tel,
        }
    loglike_fns = {
        "lin": likelihood_fn,
        "shm": likelihood_fn,
        # "norm": norm_likelihood_fn,
    }
    posterior_fns = {
        "lin": posterior_fn,
        "shm": posterior_fn,
        # "norm": norm_posterior_fn,
    }
    datas = {
        "lin": lin_datas,
        "shm": shm_datas,
        # "norm": norm_datas,
    }

    common_params = [
        "separation",
        "position_angle",
        "x_position",
        "y_position",
        "log_flux",
        "contrast",
        # "wavelengths",
        # "psf_pixel_scale",
    ]

    lin_params = [
        "jitter_mag",
        "jitter_angle",
        "aperture.coefficients",
    ]

    # norm_params = [
    #     "Jitter.r",
    #     "Jitter.shear",
    #     "Jitter.phi",
    #     "aperture.coefficients",
    # ]

    params = {
        "lin": common_params + lin_params,
        "shm": common_params + lin_params,
        # "norm": common_params + norm_params,
    }

    return models, datas, params, loglike_fns, posterior_fns

models, datas, params, loglike_fns, posterior_fns = setup_jitter()

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
    # "jitter_mag",
    "jitter_angle",
    "aperture.coefficients",
    # # 'wavelengths',
    # # 'psf_pixel_scale',
]

test_mags = np.linspace(1e-4, 2 * 0.375, 30)
test_angs = np.linspace(0, 90, 7)

# test_mags = np.linspace(1e-4, 2 * 0.375 /3, 10)
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
run_compute=False
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
        # seps_shm,
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
# plt.savefig(nt_files_path + "paper_figs/lin_shm_sweep.pdf", bbox_inches="tight", dpi=500)
plt.close()
