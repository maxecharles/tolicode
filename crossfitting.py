# %% [markdown]
# # Fisher Information Analysis and Model Cross-Fitting
#
# Code used to run analysis and generate plots for "Mitigating effects of jitter through differentiable
# forwards-modeling for the TOLIMAN space telescope".

# %%
import jax

# Enable 64bit precision (note this must be run in the first cell of the notebook)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "gpu")

from jax import numpy as np, random as jr, scipy as jsp
import zodiax as zdx
import dLux
import dLuxToliman as dlT
from dLuxToliman import AlphaCen

from tqdm import tqdm
from datetime import datetime
import secrets

# plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
import scienceplots

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
# nt_files_path = "/home/max/code/tolicode/files/"
nt_files_path = "/fred/oz440/max/code/tolicode/files/"

# %% [markdown]
# ## Priors

from jax.scipy.stats import norm, beta

Source = lambda: dLux.BaseSource
Optics = lambda: dLux.BaseOpticalSystem


from jax.scipy.stats import norm, beta
from xfitting_helpers import fwhm_to_det, NGDToliman, NGDJitteredToliman


def weibull_logpdf(x, k=1.2, lam=250):
    x = np.asarray(x)
    return np.log(k) - k * np.log(lam) + (k - 1) * np.log(x) - (x / lam) ** k


# fig, ax = plt.subplots(2, 2, figsize=(10, 5))

# xs = np.linspace(-1, 100, 2000)
# ax[0][0].fill_between(xs, np.exp(weibull_logpdf(xs)), alpha=0.5)
# ax[0][0].set(
#     title=r"Prior on determinant $r$",
#     xlabel=r"Determinant $r$ [arcsec$^4 \times10^{6}$]",
#     ylabel="Probability Density",
#     xlim=(-1, 100),
#     ylim=(0, None),
# )

# xs = np.linspace(0, 1, 2000)
# ax[0][1].fill_between(xs, beta.pdf(xs, a=1.1, b=1.1), alpha=0.5)
# ax[0][1].set(
#     title=r"Prior on shear $\eta$",
#     xlabel=r"Shear $\eta$",
#     ylabel="Probability Density",
#     ylim=(0, None),
# )

# xs = np.linspace(-4, 4, 2000)
# ax[1][0].fill_between(xs, norm.pdf(x=xs, loc=0, scale=1.0), alpha=0.5)
# ax[1][0].set(
#     title=r"Prior on jitter angle $\phi$",
#     xlabel=r"Jitter angle $\phi$ [deg]",
#     ylabel="Probability Density",
#     ylim=(0, None),
# )


# xs = np.linspace(-16, 16, 2000)
# ax[1][1].fill_between(xs, norm.pdf(x=xs, loc=0, scale=4.0), alpha=0.5)
# ax[1][1].set(
#     title=r"Prior on aberration coefficients $Z$",
#     xlabel=r"Zernike Coefficient",
#     ylabel="Probability Density",
#     ylim=(0, None),
# )
# plt.tight_layout()
# # plt.show()
# plt.savefig(nt_files_path + "test/priors.png", dpi=150)
# plt.close()


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
        if "data_key" in args.keys():
            if args["data_key"] == "mvn":
                prior += norm.logpdf(x=shear, loc=0.1, scale=0.05)
            if args["data_key"] != "mvn":
                prior += norm.logpdf(x=shear, loc=0.8, scale=0.002)

        # jitter angle phi
        angle = model.get("Jitter.phi")
    prior += norm.logpdf(x=angle, loc=args["angle"], scale=1.0)

    # aberration coefficients Z
    aberrations = model.get("aperture.coefficients")
    prior += norm.logpdf(x=aberrations, loc=0.0, scale=4.0).sum()

    return prior


angle = 0.0
mag = 0.5 * 0.375
shear = 0.1
r = fwhm_to_det(0.5 * 0.375, 0.1)
oversample = 4
mvn_osamp = 6
det_pscale = 0.375
det_npixels = 128
kernel_size = 17
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
mvn_params = {"r": r, "shear": shear, "phi": angle, "kernel_size": kernel_size}
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
mvn_optics = optics.set("oversample", mvn_osamp)

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
mvn_det = dLux.LayeredDetector(
    [
        ("Jitter", dlT.GaussianJitter(**mvn_params)),
        ("Downsample", dLux.Downsample(mvn_osamp)),
    ]
)

# creating models
lin_tel = NGDJitteredToliman(source=src, optics=optics, **lin_params).set(
    "detector", lin_det
)
shm_tel = NGDJitteredToliman(source=src, optics=optics, **shm_params).set(
    "detector", shm_det
)
mvn_tel = NGDToliman(source=src, optics=mvn_optics).set("detector", mvn_det)

common_cov_params = [
    "separation",
    "position_angle",
    "x_position",
    "y_position",
    "log_flux",
    "contrast",
    # # "wavelengths",
    # # "psf_pixel_scale",
]

lin_cov_params = [
    "jitter_mag",
    "jitter_angle",
    # "aperture.coefficients",
]

mvn_cov_params = [
    "Jitter.r",
    "Jitter.shear",
    "Jitter.phi",
    # "aperture.coefficients",
]

cov_params = {
    "lin": common_cov_params + lin_cov_params,
    "shm": common_cov_params + lin_cov_params,
    "mvn": common_cov_params + mvn_cov_params,
}

# NOTE make sure to ROUND data before passing here.
likelihood_fn = lambda model, data: jsp.stats.poisson.logpmf(data, model.jitter_model())

posterior_fn = lambda model, data, args: likelihood_fn(model, data).sum() + prior_fn(
    model, args
)

mvn_likelihood_fn = lambda model, data: jsp.stats.poisson.logpmf(data, model.model())

mvn_posterior_fn = lambda model, data, args: mvn_likelihood_fn(
    model, data
).sum() + prior_fn(model, args)

# Wrapping everything up and returning
models = {"lin": lin_tel, "shm": shm_tel, "mvn": mvn_tel}
loglike_fns = {
    "lin": likelihood_fn,
    "shm": likelihood_fn,
    "mvn": mvn_likelihood_fn,
}
posterior_fns = {
    "lin": zdx.filter_jit(posterior_fn),
    "shm": zdx.filter_jit(posterior_fn),
    "mvn": zdx.filter_jit(mvn_posterior_fn),
}

# creating simulated data at a high oversample
print("Creating simulated data grid...")
# dlin_tel = lin_tel.set(["oversample", "Downsample.kernel_size"], [8, 8])
dlin_tel = lin_tel
lin_datas = []

# dshm_tel = shm_tel.set(["oversample", "Downsample.kernel_size"], [8, 8])
dshm_tel = shm_tel
shm_datas = []

# dmvn_tel = mvn_tel.set(["oversample", "Downsample.kernel_size"], [8, 8])
dmvn_tel = mvn_tel
mvn_datas = []


@zdx.filter_jit
def calc_cov(model, params, ll_fn, *ll_args):
    return zdx.covariance_matrix(
        model,
        params,
        ll_fn,
        *ll_args,
        save_memory=True,
    )


for ang in np.linspace(0, 90, 5):
    for mag in np.linspace(0.375 / 5, 0.375 / 1, 5):

        # Setting models
        dlin_tel = dlin_tel.set("jitter_mag", mag).set("jitter_angle", ang)
        dshm_tel = dshm_tel.set("jitter_mag", mag).set("jitter_angle", ang)

        fwhm = mag
        r = fwhm_to_det(fwhm, shear=0.1)
        dmvn_tel = dmvn_tel.set("Jitter.r", r).set("Jitter.phi", ang)

        models = {"lin": dlin_tel, "shm": dshm_tel, "mvn": dmvn_tel}
        # generating data and covariances for
        # LIN
        data = dlin_tel.jitter_model()
        noisy_data = jr.poisson(jr.PRNGKey(0), data)
        cov = {
            model_key: calc_cov(
                models[model_key],
                cov_params[model_key],
                posterior_fns[model_key],
                noisy_data,
                {"angle": ang},
            )
            for model_key in models.keys()
        }
        lin_data = {
            "params": ["jitter_mag", "jitter_angle"],
            "values": [mag, ang],
            "data": data,
            "cov": cov,
        }
        lin_datas.append(lin_data)

        # SHM
        data = dshm_tel.jitter_model()
        noisy_data = jr.poisson(jr.PRNGKey(0), data)
        cov = {
            model_key: calc_cov(
                models[model_key],
                cov_params[model_key],
                posterior_fns[model_key],
                noisy_data,
                {"angle": ang},
            )
            for model_key in models.keys()
        }
        shm_data = {
            "params": ["jitter_mag", "jitter_angle"],
            "values": [mag, ang],
            "data": data,
            "cov": cov,
        }
        shm_datas.append(shm_data)

        # MVN
        data = dmvn_tel.model()
        noisy_data = jr.poisson(jr.PRNGKey(0), data)
        cov = {
            model_key: calc_cov(
                models[model_key],
                cov_params[model_key],
                posterior_fns[model_key],
                noisy_data,
                {"angle": ang},
            )
            for model_key in models.keys()
        }
        mvn_data = {
            "params": ["Jitter.r", "Jitter.phi"],
            "values": [r, ang],
            "data": data,
            "cov": cov,
        }
        mvn_datas.append(mvn_data)

datas = {
    "lin": lin_datas,
    "shm": shm_datas,
    "mvn": mvn_datas,
}


# %% [markdown]
# ## Model Cross-Fitting
#
# The code for model cross-fitting to investigate systematic model-introduced bias and a potential increase in separation error.
from xfitting_helpers import plot_losses, summarise_fit
from matplotlib import colors


def run_grad_desc(
    model,
    data,
    args,
    optimisers: dict,
    gradloss_func,
    likelihood_im_fn,
    cov_params,
    norm_fn=lambda model, args: model,
    grad_fn=lambda grads, args: grads,
    iters=100,
    plot=False,
    verbose=True,
    eps=None,  # termination condition
    suffix="",
    save_path=nt_files_path,
):
    """
    Run gradient descent on a model.
    """

    # run gradient descent
    params = list(optimisers.keys())
    optim, opt_state = zdx.get_optimiser(model, params, list(optimisers.values()))
    losses, models_out = [], []

    norm_fn = zdx.filter_jit(norm_fn)
    # grad_fn = zdx.filter_jit(grad_fn)

    if verbose:
        t = tqdm(range(iters), desc="Gradient Descent")
    else:
        t = range(iters)

    for i in t:

        # for termination condition
        last_params = model.get(params)

        loss, grads = gradloss_func(model, np.round(data), args)

        # NATURAL GRADIENTS
        grads = grads.matmul(cov_params, args["data_dict"]["cov"][args["model_key"]])
        grads = grad_fn(grads, args)

        # Updating
        updates, opt_state = optim.update(grads, opt_state)
        model = zdx.apply_updates(model, updates)
        model = norm_fn(model, args)

        models_out.append(model)
        losses.append(loss)

        if verbose:
            t.set_description("Loss: {:.6e}".format(loss))  # update the progress bar

        # Termination condition
        new_params = model.get(params)  # getting new parameters
        scaled_diffs = jax.tree.map(
            lambda x, y: np.abs((x - y) / y),
            last_params,
            new_params,
        )
        if eps is not None and i > 30:
            if np.all(
                np.array(  # if all parameters have converged
                    jax.tree.map(lambda x: np.all(x < eps), scaled_diffs)
                )
            ):
                print("Converged early")
                break

    if plot:
        plot_losses(losses, save_path, 10, suffix=suffix)
        summarise_fit(model, data, likelihood_im_fn, save_path, suffix)

        params_in = params
        for p in np.arange(0, len(params), 2):
            plt.figure(figsize=(10, 3))
            plt.subplot(1, 2, 1)

            param = params_in[p]
            param_out = np.array([m.get(param) for m in models_out])

            if param_out.size // iters > 1:
                plt.plot(range(i + 1), param_out - param_out[0], label=param)
            else:
                plt.plot(range(i + 1), param_out, label=param)
            plt.xlabel("Epoch")
            plt.legend()
            plt.ticklabel_format(style="plain", axis="both", useOffset=False)

            plt.subplot(1, 2, 2)
            if p + 1 == len(params_in):
                plt.tight_layout()
                plt.savefig(save_path + f"test/params_{p}_{suffix}.png", dpi=150)
                plt.close()
                break

            param = params_in[p + 1]
            param_out = np.array([m.get(param) for m in models_out])

            if param_out.size // iters > 1:
                plt.plot(range(i + 1), param_out - param_out[0], label=param)
            else:
                plt.plot(range(i + 1), param_out, label=param)

            plt.xlabel("Epoch")
            plt.legend()
            plt.ticklabel_format(style="plain", axis="both", useOffset=False)

            plt.tight_layout()
            plt.savefig(save_path + f"test/params_{p}_{suffix}.png", dpi=150)
            plt.close()

        m = models_out[-1]
        if isinstance(m, dlT.JitteredToliman):
            sim = m.jitter_model()
        elif isinstance(m, dlT.Toliman):
            sim = m.model()

        # plot residuals
        residual = data - sim
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Model")
        plt.imshow(sim, cmap="inferno")
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.title("Data")
        plt.imshow(data, cmap="inferno")
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.title("Residual")
        plt.imshow(residual, cmap="coolwarm", norm=colors.CenteredNorm())
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(save_path + f"test/residuals_{suffix}.png", dpi=150)
        plt.close()

    return models_out[-1]


# %%
@zdx.filter_jit
def grad_fn(grads, args={}):

    print("Compiling grad_fn...")

    # if mvn model
    data_dict, model_key, data_key, optimisers = (
        args["data_dict"],
        args["model_key"],
        args["data_key"],
        args["optimisers"].keys(),
    )

    if model_key == "mvn":

        if data_key == "mvn":
            pass

        elif data_key != "mvn":
            grads = (
                grads.multiply("Jitter.r", np.array(1e-3))
                if "Jitter.r" in optimisers
                else grads
            )
            grads = (
                grads.multiply("Jitter.shear", np.array(1e-3))
                if "Jitter.shear" in optimisers
                else grads
            )
            grads = (
                grads.multiply("Jitter.phi", np.array(1e-3))
                if "Jitter.phi" in optimisers
                else grads
            )

    elif model_key != "mvn":
        if data_key == "mvn":
            pass

        elif data_key != "mvn":
            pass
    return grads


@zdx.filter_jit
def norm_fn(model, args={}):

    print("Compiling norm_fn...")

    # if norm model
    data_dict, model_key, data_key = (
        args["data_dict"],
        args["model_key"],
        args["data_key"],
    )

    # clipping shear
    if model_key == "mvn":
        shear = model.get("Jitter.shear")
        shear = np.clip(shear, 1e-8, 1 - 1e-4)
        model = model.set("Jitter.shear", shear)

        det = model.get("Jitter.r")
        det = np.clip(det, min=1e-16, max=None)
        model = model.set("Jitter.r", det)

    return model


# %%
from zodiax.optimisation import sgd, adam

sep_dict_save_dir = nt_files_path + "results/xfit/"


sep_dict = {}
n_realisations = 25


# Gradient descent
common_optimisers = {
    "separation": sgd(1e-1, 0),
    "position_angle": sgd(1e-1, 3),
    "x_position": sgd(1e-1, 0),
    "y_position": sgd(1e-1, 2),
    "log_flux": sgd(1e-1, 0),
    "contrast": sgd(1e-1, 1),
    "aperture.coefficients": sgd(1e-4, 4),
}

lin_opts = {
    "jitter_mag": sgd(1e-1, 3),
    "jitter_angle": sgd(1e-1, 5),
}

norm_opts = {
    "Jitter.r": sgd(1e-1, 1),
    "Jitter.shear": sgd(1e-1, 5),
    "Jitter.phi": sgd(1e-1, 0),
}
# looping over models
for model_key in tqdm(models.keys(), desc="Models"):

    if model_key == "mvn":
        optimisers = {**common_optimisers, **norm_opts}
    else:
        optimisers = {**common_optimisers, **lin_opts}

    posterior_fn = posterior_fns[model_key]

    @zdx.filter_jit
    @zdx.filter_value_and_grad(list(optimisers.keys()))
    def gradloss_func(model, data, args):
        return -posterior_fn(model, data, args).sum()

    # looping over data arrays
    for data_key in tqdm(datas.keys(), desc="Data Arrays"):
        # if data_key != "lin":
        #     continue
        # if data_key != model_key:
        #     continue
        # if data_key != "mvn":
        #     continue
        # if data_key == "mvn":
        #     continue
        # if data_key != "lin":
        #     continue
        # if model_key == "shm":
        #     continue
        # if data_key == "shm":
        #     continue
        # if model_key != "lin":
        # continue
        # if model_key != "mvn":
        #     continue
        model = models[model_key]
        sep_values = np.array([], dtype=np.float64)

        # looping over noise realisations
        for i in tqdm(range(n_realisations), desc="Noise Realisations"):

            data_dict = datas[data_key][i % len(datas[data_key])]
            data = data_dict["data"]
            angle = data_dict["values"][1]

            if model_key == "mvn" and data_key != "mvn":
                mag = data_dict["values"][0]
                model = model.set("Jitter.phi", angle)
                model = model.set("Jitter.shear", np.array(0.8))
                model = model.set("Jitter.r", 0.5 * fwhm_to_det(mag, shear=0.8))
            elif model_key != "mvn" and data_key == "mvn":
                model = model.set("jitter_mag", np.array(0.01))
                model = model.set("jitter_angle", angle)
            else:
                model = model.set(data_dict["params"], data_dict["values"])

            # poisson draw!
            noisy_data = jr.poisson(jr.PRNGKey(secrets.randbits(32)), data)

            args = {
                "data_dict": data_dict,
                "model_key": model_key,
                "data_key": data_key,
                "angle": angle,
                "optimisers": optimisers,
            }

            print(
                f"Model {model_key}",
                f"Data {data_key}",
                f"Realisation {i+1}/{n_realisations}",
                data_dict["values"][0],
                data_dict["values"][1],
            )
            # RUN GRAD DESCENT
            gd_model = run_grad_desc(
                model,
                noisy_data,
                args,
                optimisers=optimisers,
                norm_fn=norm_fn,
                grad_fn=grad_fn,
                cov_params=cov_params[model_key],
                iters=500,
                gradloss_func=gradloss_func,
                likelihood_im_fn=loglike_fns[model_key],
                eps=5e-4,
                plot=True,
                suffix=f"_{model_key}_{data_key}_{i}",
            )

            sep_values = np.append(sep_values, gd_model.separation)
        sep_dict[f"{model_key}_{data_key}"] = sep_values

# saving
# current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
# save_str = current_time + f"_{n_realisations:04d}.npy"
# np.save(os.path.join(sep_dict_save_dir, save_str), sep_dict)
