import deepinv as dinv

def define_physics(inv_problem, noise_model, noise_level):
    if noise_model == "gaussian":
        noise = dinv.physics.GaussianNoise(sigma=noise_level)
    if noise_model == "poisson":
        noise = dinv.physics.PoissonNoise(gain=noise_level)
    if inv_problem == "denoising":
        physics = dinv.physics.Denoising(noise=noise)
    return physics

def choose_denoiser(name, imsize = (1, 64, 64)):
    if name == "unet":
        out = dinv.models.UNet(in_channels=imsize[0], out_channels=imsize[0])
    elif name == "drunet":
        out = dinv.models.DRUNet(in_channels=imsize[0], out_channels=imsize[0])
    elif name == "scunet":
        out = dinv.models.SCUNet(in_nc=imsize[0])
    elif name == "gsdrunet":
        out = dinv.models.GSDRUNet(in_channels=imsize[0], out_channels=imsize[0])
    elif name == "bm3d":
        out = dinv.models.BM3D()
    elif name == "dncnn":
        out = dinv.models.DnCNN(in_channels=imsize[0], out_channels=imsize[0])
    elif name == "waveletdenoiser":
        out = dinv.models.WaveletDenoiser()
    elif name == "waveletdict":
        out = dinv.models.WaveletDictDenoiser()
    elif name == "waveletdict_hard":
        out = dinv.models.WaveletDictDenoiser(non_linearity="hard")
    elif name == "waveletdict_topk":
        out = dinv.models.WaveletDictDenoiser(non_linearity="topk")
    elif name == "tgv":
        out = dinv.models.TGVDenoiser(n_it_max=10)
    elif name == "tv":
        out = dinv.models.TVDenoiser(n_it_max=10)
    elif name == "median":
        out = dinv.models.MedianFilter()
    elif name == "autoencoder":
        out = dinv.models.AutoEncoder(dim_input=imsize[0] * imsize[1] * imsize[2])
    elif name == "swinir":
        out = dinv.models.SwinIR(in_chans=imsize[0])
    elif name == "epll":
        out = dinv.models.EPLLDenoiser(channels=imsize[0])
    elif name == "restormer":
        out = dinv.models.Restormer(in_channels=imsize[0], out_channels=imsize[0])
    else:
        raise Exception("Unknown denoiser")
    return out