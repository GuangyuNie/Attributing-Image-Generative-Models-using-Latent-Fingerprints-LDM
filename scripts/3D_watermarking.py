import argparse, os, sys, glob, datetime, yaml, math

import torch
import time
import numpy as np
from tqdm import trange, tqdm
from torchmetrics.image.inception import InceptionScore


from omegaconf import OmegaConf
from PIL import Image
import matplotlib.pyplot as plt
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

from torch.nn import functional as F
import lpips_custom
import scipy.stats.qmc as scipy_stats


class Watermark():

    def load_pca(self, data):
        return data["var"], data["mean"], data["pc"]


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0, xt=None):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]

    # Sample is the latent output for ddim, do pca here
    if xt != None:
        samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False, x_T=xt)
    else:
        samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False)
    return samples, intermediates


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def custom_to_watermarking(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


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


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_subset_pc(pc, key_len=64, init=0):
    return pc[init:init + key_len], np.delete(pc, list(range(init, init + key_len, 1)), axis=0)


def get_projection_matrix(axis):
    print('getting projection matrix...')
    axis = axis.T
    projection_matrix = np.dot(np.dot(axis, np.linalg.inv(np.dot(axis.T, axis))), axis.T)
    print('projecting...')
    return projection_matrix


def projection(sample, projection_matrix):
    # project sample on to axis
    sample_projected = np.empty(np.shape(sample))

    proj_of_sample_on_ur = np.dot(sample[0], projection_matrix)
    sample_projected[0] = proj_of_sample_on_ur

    return sample_projected


def get_perturbed_latent(sample, v, proj_mean):
    # project sample on to axis
    latent = np.empty(np.shape(sample))
    key = np.random.randint(2, size=opt.key_len)
    key = np.repeat(key, 3)
    # key = opt.sigma * np.ones(opt.key_len)
    latent[0] = sample[0] + opt.sigma * np.dot(key, v) + proj_mean
    return latent, key


def get_latent_from_key(alpha, key, u, v, mean):
    # project sample on to axis
    latent = torch.empty((1, 3 * 4096), device=model.device)
    latent[0] = torch.matmul(alpha[0], u) + opt.sigma * torch.matmul(key, v) + mean

    return latent.reshape(1, 3, 64, 64)


def get_editing_direction():
    var, mean, pc = Watermark.load_pca(pca)
    shift = opt.shift
    v, u = get_subset_pc(pc, 3 * opt.key_len, shift)

    var_v, var_u = get_subset_pc(var, 3 * opt.key_len, shift)
    return v, u, mean, var_v, var_u


def get_sampling_range(proj):
    v, u, mean, _, _ = get_editing_direction()
    sample_projected = projection([mean], proj)
    mean_alpha, _ = torch.lstsq(torch.as_tensor(sample_projected[0].reshape(-1, 1), dtype=torch.float32,
                                                device=model.device), torch.as_tensor(u.T, device=model.device))

    return mean_alpha[0:3 * (4096 - opt.key_len)]


def get_minmax():
    shape = [1, 3, 64, 64]
    v, u, mean, var_v, var_u = get_editing_direction()
    std_ur = torch.sqrt(torch.as_tensor(var_u, device=model.device).reshape(-1))
    c = 3

    return c * std_ur, -c * std_ur


def get_batched_latent(proj, proj_v,v, u, mean, var_v, var_u):
    shape = [opt.batch_size, model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    samples, _ = convsample_ddim(model, opt.custom_steps, shape, opt.eta)
    [mean_p] = projection([mean], proj_v)
    latent = np.empty(shape)
    samples_np = samples.detach().cpu().numpy().reshape(opt.batch_size, 1, -1)
    key_total = np.empty([opt.batch_size, 3 * opt.key_len])
    for i in range(opt.batch_size):
        projected = projection(samples_np[i], proj)  # 3*4096
        latent_i, key = get_perturbed_latent(projected, v, mean_p)
        latent[i] = latent_i.reshape(shape[1:])
        key_total[i] = key

    return samples, latent, key_total


@torch.no_grad()
def make_convolutional_sample(model, proj, proj_v,v, u, mean, var_v, var_u,
                              batch_size, vanilla=False, custom_steps=None, eta=1.0, ):
    log = dict()
    # shape = [batch_size,
    #          model.model.diffusion_model.in_channels,
    #          model.model.diffusion_model.image_size,
    #          model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        samples, latent, key = get_batched_latent(proj, proj_v, v, u, mean, var_v, var_u )

    latent = torch.as_tensor(latent, dtype=torch.float32, device=model.device)

    # x sample is generated image
    t0 = time.time()
    perturbed_sample = model.decode_first_stage(latent)
    original_sample = model.decode_first_stage(samples)
    t1 = time.time()
    log["sample_p"] = perturbed_sample
    log["sample_o"] = original_sample
    log["time"] = t1 - t0
    log['throughput'] = latent.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log, latent, latent, key


def key_init_guess():
    """init guess for key, all zeros (before entering sigmoid function)"""
    return torch.zeros((1, opt.key_len), device=model.device)


def alpha_init_guess():
    return torch.zeros((1, 3 * 4096 - opt.key_len), device=model.device)


def calculate_classification_acc(approx_key, target_key):
    """Calculate digit-wise key classification accuracy"""
    key_acc = torch.sum(target_key == approx_key)
    acc = key_acc / (3 * opt.key_len)
    return acc


def get_loss(img1, img2, loss_func='perceptual'):
    """Loss function, default: MSE loss"""
    if loss_func == "mse":
        loss = F.mse_loss(img1, img2).float()
    elif loss_func == "perceptual":
        loss = percept(img1, img2)
    return loss


def penalty_1(latent, upper, lower):
    """penalty for alpha that exceed the boundary"""
    penalty1 = torch.sum(relu(latent - upper))
    penalty2 = torch.sum(relu(lower - latent))
    # plt.plot(latent.detach().cpu().numpy())
    # plt.plot(upper.detach().cpu().numpy())
    # plt.plot(lower.detach().cpu().numpy())
    # plt.show()

    return penalty1 + penalty2


def get_watermark(imgs, true_key, true_latent):
    v, u, mean, var_v, var_u = get_editing_direction()
    v = torch.as_tensor(v, device=model.device)
    u = torch.as_tensor(u, device=model.device)

    mean = torch.as_tensor(mean, device=model.device)
    # imgpath = os.path.join(logdir, 'steps/')
    max, min = get_minmax()
    # os.mkdir(imgpath)
    acc = 0
    total_approximation_error = torch.empty(100, 2000)
    for iter, img in enumerate(imgs):
    # target_img = imgs.clamp_(min=-1, max=1).permute(0,3,1,2).add(1).div_(2).mul(255).round().type(torch.uint8).to(model.device)
    # inception.update(target_img)
    # inception_score = inception.compute()
    # torch.save(inception_score, logdir + '/IS.pt')
        sample = samlping.random(n=opt.lhs)  # Sample init guesses
        sample = torch.tensor(sample, dtype=torch.float32, device=model.device).detach()
        true_key_i = torch.as_tensor(true_key[iter], device=model.device)
        true_latent_i = true_latent[iter]
        true_latent_i_np = true_latent_i.detach().cpu().numpy().reshape(1, -1)
        # a,b,c = get_alpha_debug(true_latent_i_np[0],true_latent_i_np[1],true_latent_i_np[2])
        # alpha = alpha_init_guess()
        img = img.permute(2, 0, 1)
        img = img[None, :].to(model.device)

        # steps_img = custom_to_pil(img[0])
        # imgpath = os.path.join(logdir, f"steps/true_{i:06}.png")
        # steps_img.save(imgpath)
        std_u = torch.sqrt(torch.as_tensor(var_u.reshape(1, -1), device=model.device))

        for alpha in sample:
            alpha = alpha.reshape(1, 3 * (4096 - opt.key_len))
            alpha[0] = 2 * torch.multiply(alpha[0], std_u)

            # alpha[0] = a.reshape(1,-1) - mean_alpha_r.reshape(1,-1)
            # alpha[1] = b.reshape(1,-1) - mean_alpha_g.reshape(1,-1)
            # alpha[2] = c.reshape(1,-1) - mean_alpha_b.reshape(1,-1)

            alpha.requires_grad = True
            key = key_init_guess()
            key.requires_grad = True
            optimizer = torch.optim.Adam([alpha, key], lr=opt.lr)
            for i in tqdm(range(opt.steps)):
                model.zero_grad()
                optimizer.zero_grad()
                key_3D = key.repeat_interleave(3)
                estimated_latent = get_latent_from_key(alpha, sigmoid(key_3D), u, v, mean)

                # approximation_error = (torch.div(torch.abs(true_latent_i.reshape(-1) - estimated_latent.reshape(-1)),
                #                                  torch.abs(true_latent_i.reshape(-1))))
                # mean = torch.mean(approximation_error)
                # total_approximation_error[iter][i] = approximation_error[0]

                estimated_image = model.differentiable_decode_first_stage(estimated_latent).to(model.device)  # 0.08s
                loss_1 = get_loss(img, estimated_image, loss_func="perceptual") + \
                         opt.lam * (penalty_1(alpha[0], max, min))
                # loss_1 = get_loss(img, estimated_image, loss_func="perceptual")
                decay = 0.001
                lr = opt.lr * math.exp(-decay * (i + 1))
                optimizer.param_groups[0]["lr"] = lr
                loss_1.backward()
                optimizer.step()

                if (i) % 100 == 0:
                    print("Perceptual loss: {:.6f}".format(loss_1.item()))
                    print('key_acc: {:.6f}'.format(cos(torch.round(sigmoid(key_3D)), true_key_i)))
                    print('latent cosine similarity: {:.6f}'.format(
                        cos(estimated_latent[0].reshape(-1), true_latent_i.reshape(-1))))
                    steps_img = custom_to_pil(estimated_image[0])
                    # imgpath = os.path.join(logdir, f"steps/{i:06}.png")
                    # steps_img.save(imgpath)

            acc = calculate_classification_acc(torch.round(sigmoid(key_3D)), true_key_i)
            torch.save(total_approximation_error, 'avg_error.pt')
            if acc == 1:
                break
    return acc


def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir, '*.png'))) - 1
    # path = logdir
    inception = InceptionScore().to(model.device)

    if model.cond_stage_model is None:
        perturbed_images = []
        original_images = []
        watermarked_images = []
        acc_total = []
        v, u, mean, var_v, var_u = get_editing_direction()
        proj = get_projection_matrix(u)
        proj_v = get_projection_matrix(v)
        # mean_alpha_r, mean_alpha_g, mean_alpha_b = get_sampling_range(proj_r, proj_g, proj_b)
        print(f"Running unconditional sampling for {n_samples} samples")
        success = 0

        for iter in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs, original_latent, new_latent, true_key = make_convolutional_sample(model, proj,
                                                                                    proj_v,v, u, mean, var_v, var_u,
                                                                                    batch_size=batch_size,
                                                                                    vanilla=vanilla,
                                                                                    custom_steps=custom_steps, eta=eta)
            # imgs = custom_to_watermarking(logs["sample_p"])
            # target_img = imgs.clamp_(min=-1, max=1).permute(0, 3, 1, 2).add(1).div_(2).mul(255).round().type(
            #     torch.uint8).to(model.device)
            # inception.update(target_img)
            # inception_score = inception.compute()
            # torch.save(inception_score, logdir + '/IS.pt')
            n_saved_1 = save_logs(logs, logdir, n_saved=n_saved, key="sample_p")
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample_o")
            perturbed_images.extend([custom_to_np(logs["sample_p"])])
            original_images.extend([custom_to_np(logs["sample_o"])])
            watermarked_images.extend([custom_to_watermarking(logs["sample_p"])])
            acc = get_watermark(custom_to_watermarking(logs["sample_p"]), true_key, new_latent)
            if opt.steps != 1:
                acc = get_watermark(custom_to_watermarking(logs["sample_p"]), true_key, new_latent)
                print(acc)
                acc_total.append(acc)
                if acc == 1.0:
                    success += 1
                classification_acc = success / (iter + 1)
                if n_saved >= n_samples:
                    print(f'Finish after generating {n_saved} samples')
                    break
                with open(logdir + 'result.txt', 'w') as filehandle:
                    for i, listitem in enumerate(acc_total):
                        filehandle.write('\n sample index: {}, key acc: {}, success rate {}'.format(i, listitem.item(),
                                                                                                    classification_acc))
        all_img = np.concatenate(perturbed_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

        all_img = np.concatenate(original_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
        raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling fof {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")

    return all_img


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key_len",
        type=int,
        default=64,
        help="key_len",
    )
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
        default=1000
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
        default=1
    )
    parser.add_argument(
        "--sigma",
        type=float,
        nargs="?",
        help="editing strength",
        default=1
    )
    parser.add_argument(
        "--lr",
        type=float,
        nargs="?",
        help="learning rate for watermark retrieval",
        default=0.2
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="?",
        help="steps for watermark retrieval",
        default=1
    )
    parser.add_argument(
        "--device_id",
        type=int,
        nargs="?",
        help="GPU device id",
        default=0
    )
    parser.add_argument(
        "--lhs",
        type=int,
        nargs="?",
        help="latin hyper cube number of samples",
        default=20
    )
    parser.add_argument(
        "--lam",
        type=float,
        help="lambda for alpha bound",
        default=0.005
    )
    parser.add_argument(
        "--shift",
        type=int,
        help="editing direction starting index",
        default=4000
    )
    return parser


if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # get time
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    # load parser
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    # load models
    Watermark = Watermark()

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
    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    print(f"global step: {global_step}")
    print(75 * "=")
    # making saving dir
    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")
    pca = np.load('./pca_1D.npy', allow_pickle=True).item()

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)
    percept = lpips_custom.PerceptualLoss(model="net-lin", net="vgg", use_gpu=model.device, gpu_ids=[opt.device_id])
    sigmoid = torch.nn.Sigmoid()
    relu = torch.nn.ReLU()
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    samlping = scipy_stats.LatinHypercube(d=(3 * (4096 - opt.key_len)), centered=True)

    all_image = run(model, imglogdir, eta=opt.eta,
                    vanilla=opt.vanilla_sample, n_samples=opt.n_samples, custom_steps=opt.custom_steps,
                    batch_size=opt.batch_size, nplog=numpylogdir)

    print("done.")










