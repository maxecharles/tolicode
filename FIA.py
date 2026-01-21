"""
SCRIPT TO RUN FISHER INFORMATION ANALYSIS ON LINEAR AND SHM JITTER MODELS
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


################## MODEL SETUP ##################

angle = 0.0  # initialise at 0 degrees, but this doesn't matter
mag = 0.5 * 0.375  # initialise at half a pixel, but this doesn't matter
oversample = 4
det_pscale = 0.375  # arcsec/pixel
det_npixels = 128
n_psfs = 5

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

# creating telescopes
lin_det = dLux.LayeredDetector([("Downsample", dLux.Downsample(oversample))])
shm_det = lin_det

# creating models
lin_tel = dlT.JitteredToliman(source=src, optics=optics, **lin_params).set(
    "detector", lin_det
)
shm_tel = dlT.JitteredToliman(source=src, optics=optics, **shm_params).set(
    "detector", shm_det
)


# NOTE make sure to ROUND data before passing here.
@zdx.filter_jit
def likelihood_fn(model, data):
    ll = jsp.stats.poisson.logpmf(data, model.jitter_model())
    return ll.sum()


cov_mat = zdx.filter_jit(zdx.covariance_matrix)


def fisher_sweep(
    tel,
    ll_fn,
    params,
    mags=np.linspace(1e-4, 2 * 0.375, 10),
    angs=np.linspace(0, 90, 3),
    save_memory=False,
):
    """
    Sweeps over different jitter magnitudes and angles, calculating the
    Fisher Information Analysis covariance matrix at each point.
    """

    seps = []

    # looping over jitter magnitude
    for mag_idx, mag in tqdm(enumerate(mags), total=len(mags)):
        model = tel.set("jitter_mag", mag)

        # looping over jitter angle
        for ang_idx, ang in enumerate(angs):
            model = model.set("jitter_angle", ang)
            data = model.jitter_model()

            # calculate covariance matrix
            cov = cov_mat(model, params, ll_fn, np.round(data), save_memory=save_memory)

            # read separation from covariance matrix
            sep = np.sqrt(np.abs(cov[0, 0]))
            seps.append(sep)

            # check for NaNs
            if np.isnan(sep).any():
                print(f"NaNs found at mag_idx={mag_idx}, ang_idx={ang_idx}")
                raise ValueError

    seps = np.array(seps).reshape(len(mags), len(angs))
    return seps


# Marginal parameters
# Ignore covariant wavelengths and pixel scale for now
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

# Points to test
test_mags = np.linspace(1e-4, 2 * 0.375, 60)
test_angs = np.linspace(0, 90, 7)

# saving test points
np.save(nt_files_path + "results/lin/test_mags.npy", test_mags)
np.save(nt_files_path + "results/shm/test_mags.npy", test_mags)
np.save(nt_files_path + "results/lin/test_angs.npy", test_angs)
np.save(nt_files_path + "results/shm/test_angs.npy", test_angs)

################## CALCULATIONS ##################

seps_lin = fisher_sweep(
    lin_tel,
    likelihood_fn,
    params,
    mags=test_mags,
    angs=test_angs,
    save_memory=save_ram,
)
np.save(nt_files_path + "results/lin/fia_seps.npy", seps_lin)

seps_shm = fisher_sweep(
    shm_tel,
    likelihood_fn,
    params,
    mags=test_mags,
    angs=test_angs,
    save_memory=save_ram,
)
np.save(nt_files_path + "results/shm/fia_seps.npy", seps_shm)


# calculate the no jitter baseline
baseline_params = [
    "separation",
    "position_angle",
    "x_position",
    "y_position",
    "log_flux",
    "contrast",
    # "jitter_mag",
    # "jitter_angle",
    "aperture.coefficients",
    # # 'wavelengths',
    # # 'psf_pixel_scale',
]
stable_model = lin_tel.set(
    "jitter_mag", np.array(0.0)
)  # this shouldn't be necessary since we only call .model but JIC
stable_data = stable_model.model()
stable_cov = cov_mat(
    stable_model,
    baseline_params,
    likelihood_fn,
    np.round(stable_data),
    save_memory=save_ram,
)
baseline = np.sqrt(np.abs(stable_cov[0, 0]))
print(f"Baseline with no wavelengths or pixel scale: {1000 * baseline} mas")
np.save(nt_files_path + "results/baseline.npy", 1000 * baseline)
np.save(nt_files_path + "results/matrices/covariance.npy", stable_cov)
np.save(nt_files_path + "results/matrices/fisher.npy", np.linalg.inv(stable_cov))
