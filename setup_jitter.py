import jax
from jax import numpy as np, scipy as jsp, Array

jax.config.update("jax_enable_x64", True)

import zodiax as zdx
from zodiax import filter_vmap

import dLuxToliman as dlT
import dLux
from dLuxToliman import AlphaCen

Source = lambda: dLux.BaseSource
Optics = lambda: dLux.BaseOpticalSystem


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


class AlphaCenMeanWavel(AlphaCen):
    n_wavels: int
    mean_wavelength: float

    def __init__(
        self,
        n_wavels=3,
        separation=10.0,  # arcseconds
        position_angle=90,  # degrees
        x_position=0.0,  # arcseconds
        y_position=0.0,  # arcseconds
        log_flux=6.832,  # Photons
        contrast=3.37,
        bandpass=(530, 640),  # nm
        weights=None,
    ):
        super().__init__(
            n_wavels=n_wavels,
            separation=separation,
            position_angle=position_angle,
            x_position=x_position,
            y_position=y_position,
            log_flux=log_flux,
            contrast=contrast,
            bandpass=bandpass,
            weights=weights,
        )

        self.n_wavels = n_wavels
        self.mean_wavelength = np.array(bandpass).mean()  # nm

    # @property
    # def wavelengths(self) -> Array:
    #     """
    #     The wavelengths of the bandpass in meters.
    #     """
    #     wavelengths = np.linspace(self.bandpass[0], self.bandpass[1], self.n_wavels)
    #     return 1e-9 * (wavelengths - wavelengths.mean() + self.mean_wavelength)

    def model(
        self: Source(),
        optics: Optics(),
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:
        """
        Models the PSF by propagating the AlphaCen source through the given optics.
        The chromatic PSF is modelled around the mean wavelength of the bandpass.

        Parameters
        ----------
        optics : Optics
            The optics to propagate the source through.
        return_wf
            Whether or not to return the wavefront.
        return_psf
            Whether or not to return the PSF.

        Returns
        -------
        psf : Array
            The PSF of the source modelled through the optics.
        """
        # Get Values
        weights = self.norm_weights
        fluxes = self.raw_fluxes
        positions = self.xy_positions

        # Here using the mean wavelength to get wavelength array
        wavelengths = np.linspace(self.bandpass[0], self.bandpass[1], self.n_wavels)
        wavelengths = 1e-9 * (wavelengths - wavelengths.mean() + self.mean_wavelength)

        # Model PSF
        input_weights = weights * fluxes[:, None]

        # Return wf case is simple
        prop_fn = lambda position, weight: optics.propagate(
            wavelengths, position, weight, return_wf, return_psf
        )
        output = filter_vmap(prop_fn)(positions, input_weights)

        # Return wf is simple case
        if return_wf:
            return output

        # Return psf just requires constructing object
        if return_psf:
            return dLux.PSF(output.data.sum(0), output.pixel_scale.mean())

        # Return array is simple
        return output.sum(0)


def setup_jitter(
    angle=0.0,
    mag=0.5 * 0.375,
    shear=0.1,
    r=fwhm_to_det(0.5 * 0.375, 0.1),
    oversample=4,
    norm_osamp=6,
    det_pscale=0.375,
    det_npixels=128,
    kernel_size=17,
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
    norm_params = {"r": r, "shear": shear, "phi": angle, "kernel_size": kernel_size}
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
    norm_optics = optics.set("oversample", norm_osamp)

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
    norm_det = dLux.LayeredDetector(
        [
            ("Jitter", dlT.GaussianJitter(**norm_params)),
            ("Downsample", dLux.Downsample(norm_osamp)),
        ]
    )

    # creating models
    lin_tel = dlT.JitteredToliman(source=src, optics=optics, **lin_params).set(
        "detector", lin_det
    )
    shm_tel = dlT.JitteredToliman(source=src, optics=optics, **shm_params).set(
        "detector", shm_det
    )
    norm_tel = dlT.Toliman(source=src, optics=norm_optics).set("detector", norm_det)

    # creating simulated data at a high oversample
    print("Creating simulated data grid...")
    # dlin_tel = lin_tel.set(["oversample", "Downsample.kernel_size"], [8, 8])
    dlin_tel = lin_tel
    lin_datas = []

    # dshm_tel = shm_tel.set(["oversample", "Downsample.kernel_size"], [8, 8])
    dshm_tel = shm_tel
    shm_datas = []

    # dnorm_tel = norm_tel.set(["oversample", "Downsample.kernel_size"], [8, 8])
    dnorm_tel = norm_tel
    norm_datas = []

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

        for r in np.linspace(
            fwhm_to_det(0.375 / 5, 0.1), fwhm_to_det(0.375 / 1, 0.1), 5
        ):
            dnorm_tel = dnorm_tel.set("Jitter.r", r).set("Jitter.phi", ang)
            norm_data = {
                "params": ["Jitter.r", "Jitter.phi"],
                "values": [r, ang],
                "data": dnorm_tel.model(),
            }
            norm_datas.append(norm_data)

    likelihood_fn = lambda model, data: jsp.stats.poisson.logpmf(
        np.round(data), model.jitter_model()
    ).sum()

    posterior_fn = lambda model, data, args: likelihood_fn(model, data) + prior_fn(
        model, args
    )

    norm_likelihood_fn = lambda model, data: jsp.stats.poisson.logpmf(
        np.round(data), model.model()
    ).sum()

    norm_posterior_fn = lambda model, data, args: norm_likelihood_fn(
        model, data
    ) + prior_fn(model, args)

    # Wrapping everything up and returning
    models = {"lin": lin_tel, "shm": shm_tel, "norm": norm_tel}
    loglike_fns = {
        "lin": likelihood_fn,
        "shm": likelihood_fn,
        "norm": norm_likelihood_fn,
    }
    posterior_fns = {
        "lin": posterior_fn,
        "shm": posterior_fn,
        "norm": norm_posterior_fn,
    }
    datas = {
        "lin": lin_datas,
        "shm": shm_datas,
        "norm": norm_datas,
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

    norm_params = [
        "Jitter.r",
        "Jitter.shear",
        "Jitter.phi",
        "aperture.coefficients",
    ]

    params = {
        "lin": common_params + lin_params,
        "shm": common_params + lin_params,
        "norm": common_params + norm_params,
    }

    return models, datas, params, loglike_fns, posterior_fns


import matplotlib.pyplot as plt
from matplotlib import colormaps, colors


def plot_losses(losses, start, stop=-1):
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
    plt.show()


def summarise_fit(
    model,
    data,
    loglike_fn,
):

    inferno = colormaps["inferno"]
    seismic = colormaps["seismic"]
    inferno.set_bad("k", 0.5)
    seismic.set_bad("k", 0.5)

    sim = model.model()
    residual = data - sim

    loglike_im = loglike_fn(model, data)
    final_loss = np.nanmean(-loglike_im)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"Pixel neg log posterior: {final_loss:,.1f}")
    plt.imshow(-loglike_im, cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Mean noise normalised slope residual")
    plt.imshow(residual, cmap=seismic, norm=colors.CenteredNorm())
    plt.colorbar()

    plt.tight_layout()
    plt.show()
