"""
SCRIPT TO RUN FISHER INFORMATION ANALYSIS ON MULTIVARIATE NORMAL JITTER MODEL
"""

import jax

# Enable 64bit precision (note this must be run in the first cell of the notebook)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

print(jax.devices())
print(jax.config.jax_enable_x64)

from jax import numpy as np, scipy as jsp
import zodiax as zdx
import dLuxToliman as dlT
import dLux
from tqdm import tqdm

save_ram = False
nt_files_path = "/fred/oz440/max/code/tolicode/files/"


def fwhm_to_det(fwhm, shear):
    return 1e6 * (1 - shear) ** 2 * (fwhm / 2.35482) ** 4


def det_to_fwhm(det, shear):
    return 2.35482 * ((det / 1e6) / (1 - shear) ** 2) ** 0.25


def powspace(start, stop, power, num):
    """
    To generate r values at appropriate intervals.
    """
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


################## MODEL SETUP ##################

angle = 0.0  # initialise at 0 degrees, but this doesn't matter
r = fwhm_to_det(0.5 * 0.375, 0.1)  # initialise at half a pixel, but this doesn't matter
shear = 0.1
kernel_size = 17
oversample = 4
det_pscale = 0.375  # arcsec/pixel
det_npixels = 128
n_psfs = 5

mvn_params = {
    "r": r,
    "shear": shear,
    "phi": angle,
    "kernel_size": kernel_size,
}

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

# Creating common source
src = dlT.AlphaCen(
    separation=np.array(10.0),
    position_angle=np.array(90.0),
    x_position=np.array(0.0),
    y_position=np.array(0.0),
    log_flux=np.array(7.581),
    contrast=np.array(3.37),
)

# creating telescope
det = dLux.LayeredDetector(
    [
        ("Jitter", dlT.GaussianJitter(**mvn_params)),
        ("Downsample", dLux.Downsample(oversample)),
    ]
)

# creating models
tel = dlT.Toliman(source=src, optics=optics).set("detector", det)


# NOTE make sure to ROUND data before passing here.
@zdx.filter_jit
def likelihood_fn(model, data):
    ll = jsp.stats.poisson.logpmf(data, model.model())
    return ll.sum()


cov_mat = zdx.filter_jit(zdx.covariance_matrix)


# Marginal parameters
# Ignore covariant wavelengths and pixel scale for now
params = [
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
    # # 'wavelengths',
    # # 'psf_pixel_scale',
]

# Points to test
# test_phis = np.linspace(0, 90, 3)
test_phis = np.linspace(0, 90, 7)
test_shears = np.array([0, 0.3])
test_shears = np.array([0, 0.3, 0.7])
test_rs = powspace(
    fwhm_to_det(1e-2 * det_pscale, test_shears[0]),
    fwhm_to_det(1.01 * det_pscale, test_shears[0]),
    2,
    60,
)

# saving test points
np.save(nt_files_path + "results/lin/test_rs.npy", test_rs)
np.save(nt_files_path + "results/shm/test_shears.npy", test_shears)
np.save(nt_files_path + "results/lin/test_phis.npy", test_phis)

################## CALCULATIONS ##################

seps = []
fwhms = []
kernels = []

for shear_idx, shear in enumerate(test_shears):
    model = tel.set("detector.Jitter.shear", shear)

    for r_idx, r in tqdm(enumerate(test_rs), total=len(test_rs)):
        fwhm = det_to_fwhm(r, shear)
        fwhms.append(fwhm)
        model = model.set("detector.Jitter.r", r)

        for phi_idx, phi in enumerate(test_phis):

            # skipping over different angles for shear = 0 as symmetric
            if shear_idx == 0 and phi_idx != 0:
                sep = np.nan

            else:
                model = model.set("detector.Jitter.phi", phi)
                data = model.model()

                cov = cov_mat(
                    model,
                    params,
                    likelihood_fn,
                    np.round(data),
                    save_memory=False,
                )

                sep = np.sqrt(np.abs(cov[0, 0]))
                if phi_idx == 0:
                    if r == test_rs.max():
                        kernels.append(
                            model.Jitter.generate_kernel(det_pscale / oversample)
                        )

            seps.append(sep)

seps = np.array(seps).reshape(len(test_shears), len(test_rs), len(test_phis))
fwhms = np.array(fwhms).reshape(len(test_shears), len(test_rs))

# saving
np.save(nt_files_path + "results/mvn/fia_seps.npy", seps)
np.save(nt_files_path + "results/mvn/kernels.npy", kernels)
np.save(nt_files_path + "results/mvn/fwhms.npy", fwhms)


import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

baseline = None

cmap = mpl.colormaps["cmr.gem"]
sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=90))
colors = cmap(test_phis / 90)

fig, axes = plt.subplots(
    len(test_shears), 2, figsize=(9, 7), sharey="col", layout="compressed"
)

for i, shear in enumerate(test_shears):
    ax, axe = axes[i]
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 2))
    ax.tick_params(axis="x", which="both", top=False)

    for sep, phi, c in zip(seps[i].T, test_phis, colors):
        if shear == 0:
            label = None
            c = "k"
        else:
            label = label = r"$\phi\,=$" + f" {phi:.0f}"
        ax.plot(test_rs, 1000 * sep, label=label, color=c, marker=None, linewidth=2.75)
    ax.set(
        xlabel=r"det$\,\Sigma$ [arcsec$^{4}$]",
        ylabel=r"Separation Error $\sigma$ [mas]",
        # ylim=(baseline - 0.01, 0.8),
        xlim=(0, test_rs.max()),
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
    # ax.axhline(
    #     baseline, linestyle="--", c="k", alpha=0.4, label="No Jitter", linewidth=1
    # )

    if shear == 0:
        cbar = fig.colorbar(sm, ax=ax)
        cbar.ax.set_visible(False)  # Hide the colorbar visually
    else:
        cbar = fig.colorbar(sm, ax=ax, ticks=test_phis, label=r"$\phi$ [deg]")
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

    max_fwhm = det_to_fwhm(test_rs.max(), shear)
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
        lambda r: det_to_fwhm(r, test_shears[0]),
        lambda fwhm: fwhm_to_det(fwhm, test_shears[0]),
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
        lambda r: det_to_fwhm(r, test_shears[1]),
        lambda fwhm: fwhm_to_det(fwhm, test_shears[1]),
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
        lambda r: det_to_fwhm(r, test_shears[2]),
        lambda fwhm: fwhm_to_det(fwhm, test_shears[2]),
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

plt.savefig(nt_files_path + "/paper_figs/norm_sweep.png", bbox_inches="tight", dpi=500)
plt.savefig(nt_files_path + "/paper_figs/norm_sweep.pdf", bbox_inches="tight", dpi=500)
plt.close()
