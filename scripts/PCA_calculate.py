import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange,tqdm

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

rescale = lambda x: (x + 1.) / 2.


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):
    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]

    # Sample is the latent output for ddim, do pca here
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False, )
    return samples, intermediates

@torch.no_grad()
def pca_ddim(model, steps, batch_size, pca_samples, eta=1.0):
    pca = PCA()
    ddim = DDIMSampler(model)
    bs = batch_size
    shape = [model.model.diffusion_model.in_channels,model.model.diffusion_model.image_size,model.model.diffusion_model.image_size]
    latent_out_r = torch.empty([0,model.model.diffusion_model.image_size**2]).to(model.device)
    latent_out_g = torch.empty([0,model.model.diffusion_model.image_size**2]).to(model.device)
    latent_out_b = torch.empty([0,model.model.diffusion_model.image_size**2]).to(model.device)
    # Sample is the latent output for ddim, do pca here
    for _ in tqdm(range(pca_samples // batch_size)):
        latent, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False, )
        latent_out_r = torch.cat((latent_out_r,latent[:,0,:,:].reshape(bs,model.model.diffusion_model.image_size**2)),0)
        latent_out_g = torch.cat((latent_out_g,latent[:,1,:,:].reshape(bs,model.model.diffusion_model.image_size**2)),0)
        latent_out_b = torch.cat((latent_out_b,latent[:,2,:,:].reshape(bs,model.model.diffusion_model.image_size**2)),0)


    latent_out_r = latent_out_r.detach().cpu().numpy()
    latent_out_g = latent_out_g.detach().cpu().numpy()
    latent_out_b = latent_out_b.detach().cpu().numpy()
    pca.fit(latent_out_r)  # do pca for the style vector data distribution
    var_r = pca.explained_variance_  # get variance along each pc axis ranked from high to low
    pc_r = pca.components_  # get the pc ranked from high var to low var
    mean_r = latent_out_r.mean(0)

    pca.fit(latent_out_g)  # do pca for the style vector data distribution
    var_g = pca.explained_variance_  # get variance along each pc axis ranked from high to low
    pc_g = pca.components_  # get the pc ranked from high var to low var
    mean_g = latent_out_g.mean(0)

    pca.fit(latent_out_b)  # do pca for the style vector data distribution
    var_b = pca.explained_variance_  # get variance along each pc axis ranked from high to low
    pc_b = pca.components_  # get the pc ranked from high var to low var
    mean_b = latent_out_b.mean(0)

    pca = {'var_r':var_r,'pc_r':pc_r,'mean_r':mean_r,'var_g':var_g,'pc_g':pc_g,'mean_g':mean_g,'var_b':var_b,'pc_b':pc_b,'mean_b':mean_b}
    np.save("./pca.npy", pca)
    return pca

@torch.no_grad()
def pca_ddim_no_rgb(model, steps, batch_size, pca_samples, eta=1.0):
    pca = PCA()
    ddim = DDIMSampler(model)
    bs = batch_size
    shape = [model.model.diffusion_model.in_channels,model.model.diffusion_model.image_size,model.model.diffusion_model.image_size]
    latent_out = torch.empty([0,3*model.model.diffusion_model.image_size**2]).to(model.device)
    # Sample is the latent output for ddim, do pca here
    for _ in tqdm(range(pca_samples // batch_size)):
        latent, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False, )
        latent_out = torch.cat((latent_out,latent.reshape(bs,-1)),0)

    latent_out = latent_out.detach().cpu().numpy()
    pca.fit(latent_out)  # do pca for the style vector data distribution
    var = pca.explained_variance_  # get variance along each pc axis ranked from high to low
    pc = pca.components_  # get the pc ranked from high var to low var
    mean = latent_out.mean(0)

    pca = {'var':var,'pc':pc,'mean':mean}
    np.save("./pca_1D.npy", pca)
    return pca

def procrustes_rotation(X1, X2):
    mtx1, mtx2, disparity = procrustes(X1, X2)
    return mtx2

def pca_ddim_no_rgb_boostrapping(model, steps, batch_size, pca_samples, eta=1.0):
    # Perform initial PCA on full dataset
    pca = PCA()
    ddim = DDIMSampler(model)
    bs = batch_size
    shape = [model.model.diffusion_model.in_channels,model.model.diffusion_model.image_size,model.model.diffusion_model.image_size]
    latent_out = torch.empty([0,3*model.model.diffusion_model.image_size**2]).to(model.device)
    # Sample is the latent output for ddim, do pca here
    for _ in tqdm(range(pca_samples // batch_size)):
        latent, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False, )
        latent_out = torch.cat((latent_out,latent.reshape(bs,-1)),0)

    latent_out = latent_out.detach().cpu().numpy()
    pca.fit(latent_out)  # do pca for the style vector data distribution

    # Align principal components using Procrustes rotation
    for i in range(10):
        for _ in tqdm(range(pca_samples // batch_size)):
            latent, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False, )
            latent_out = torch.cat((latent_out, latent.reshape(bs, -1)), 0)

        latent_out = latent_out.detach().cpu().numpy()
        pca_sample = PCA()
        pca_sample.fit(latent_out)  # do pca for the style vector data distribution


        # Align principal components using Procrustes rotation
        mtx = procrustes_rotation(pca.components_, pca_sample.components_)
        pca.components_ = mtx

        # Store bootstrapped principal components
        if i == 0:
            boot_pca = pca.components_
        else:
            boot_pca = np.dstack((boot_pca, pca.components_))

    # Compute mean and standard deviation of bootstrapped principal components
    mean_pca = np.mean(boot_pca, axis=2)
    std_pca = np.std(boot_pca, axis=2)

    print(np.mean(std_pca))



@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0, ):
    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model, steps=custom_steps, shape=shape,
                                                    eta=eta)

        t1 = time.time()

    # x sample is generated image
    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log


def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir, '*.png'))) - 1
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
        raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )

    parser.add_argument(
        "--pca_samples",
        type=int,
        nargs="?",
        help="data samples for pca",
        default=50000
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # get time
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    # load parser
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    # load models
    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    # if argparse not provided, load config.yaml to get arg
    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    # print logdir
    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    # get model and steps trained
    model, global_step = load_model(config, ckpt, gpu, eval_mode)

    #pca_ddim_no_rgb(model, opt.custom_steps, opt.batch_size, opt.pca_samples)
    pca_ddim_no_rgb_boostrapping(model,opt.custom_steps, opt.batch_size, opt.pca_samples)

    print("done.")
