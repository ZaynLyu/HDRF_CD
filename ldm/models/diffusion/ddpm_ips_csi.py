# add at 20230704
#

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager, nullcontext
from functools import partial
import itertools
from tqdm import tqdm
from torchvision.utils import make_grid
#from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from omegaconf import ListConfig

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
import os

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 learning_rate=1.5e-05,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,# check point path
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 make_it_fit=False,
                 ucg_training=None,
                 reset_ema=False,
                 reset_num_ema_updates=False,
                 ):
        super().__init__()
        self.learning_rate=learning_rate
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")


        self.cond_stage_model = None#！！！这么重要就这么过去了？希望在后面的model 里面看到对这个的使用。
        self.clip_denoised = clip_denoised#True 应该是logvar 之类的限制范围吧
        self.log_every_t = log_every_t#100
        self.first_stage_key = first_stage_key#image 表示处理的是图片
        self.image_size = image_size  # try conv?我白银你问我？
        self.channels = channels#+3
        self.use_positional_encodings = use_positional_encodings#False

        self.model = DiffusionWrapper(unet_config, conditioning_key)#unet_config+None
        #here
        count_params(self.model, verbose=True)#计算参数数量并且print  我也想看看这么一个unet 会是多少个参数捏？
        self.use_ema = use_ema#被置为false了
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")



        #不cosine 退火
        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:#False  过
            self.scheduler_config = scheduler_config
        self.v_posterior = v_posterior#意义不明，再继续看#用在
        #posterior_variance 里面了，设置为0，就是正常的ddpm了，不影响计算了
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight



        if monitor is not None:
            self.monitor = monitor#"val/loss"
        self.make_it_fit = make_it_fit#False
        #here

        if reset_ema: assert exists(ckpt_path)#False
        if ckpt_path is not None:#现在是none 之后得保存吧，假设有
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
            if reset_ema:
                assert self.use_ema
                print(f"Resetting ema to pure model weights. This is useful when restoring from an ema-only checkpoint.")
                self.model_ema = LitEma(self.model)
        if reset_num_ema_updates:#False
            print(" +++++++++++ WARNING: RESETTING NUM_EMA UPDATES TO ZERO +++++++++++ ")
            assert self.use_ema
            self.model_ema.reset_num_updates()
        #下面定义了 参数就是那些参数
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type#l2

        self.learn_logvar = learn_logvar
        #初始为0
        logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        else:
            self.register_buffer('logvar', logvar)

        self.ucg_training = ucg_training or dict()#【】空
        if self.ucg_training:#False
            self.ucg_prng = np.random.RandomState()
    #1-3
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'


        to_torch = partial(torch.tensor, dtype=torch.float32)
        #
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))#DDPM  e4
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))#DDPM  e4
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))#log_variance
        #predict_start_from_noise #Get the distribution p(x_0 | x_t). DDPM e9变形 e10 是结合e7得到的
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))

        #这两个系数有用的计算前一个时刻的的distribution mean的。DDPM e7
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")
    @torch.no_grad()
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        if self.make_it_fit:
            n_params = len([name for name, _ in
                            itertools.chain(self.named_parameters(),
                                            self.named_buffers())])
            for name, param in tqdm(
                    itertools.chain(self.named_parameters(),
                                    self.named_buffers()),
                    desc="Fitting old weights to new weights",
                    total=n_params
            ):
                if not name in sd:
                    continue
                old_shape = sd[name].shape
                new_shape = param.shape
                assert len(old_shape) == len(new_shape)
                if len(new_shape) > 2:
                    # we only modify first two axes
                    assert new_shape[2:] == old_shape[2:]
                # assumes first axis corresponds to output dim
                if not new_shape == old_shape:
                    new_param = param.clone()
                    old_param = sd[name]
                    if len(new_shape) == 1:
                        for i in range(new_param.shape[0]):
                            new_param[i] = old_param[i % old_shape[0]]
                    elif len(new_shape) >= 2:
                        for i in range(new_param.shape[0]):
                            for j in range(new_param.shape[1]):
                                new_param[i, j] = old_param[i % old_shape[0], j % old_shape[1]]

                        n_used_old = torch.ones(old_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_old[j % old_shape[1]] += 1
                        n_used_new = torch.zeros(new_shape[1])
                        for j in range(new_param.shape[1]):
                            n_used_new[j] = n_used_old[j % old_shape[1]]

                        n_used_new = n_used_new[None, :]
                        while len(n_used_new.shape) < len(new_shape):
                            n_used_new = n_used_new.unsqueeze(-1)
                        new_param /= n_used_new

                    sd[name] = new_param

        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys:\n {missing}")
        if len(unexpected) > 0:
            print(f"\nUnexpected Keys:\n {unexpected}")
    #4-5 distribution q(x_t | x_0). p(x_0 | x_t).
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    def predict_start_from_noise(self, x_t, t, noise):
        #Get the distribution p(x_0 | x_t).
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    #6-7 有很大质疑，你真tm 用了吗？
    def predict_start_from_z_and_v(self, x_t, t, v):
        #Get the distribution p(x_0 | z,v).？有点问题，你前面系数对吗？DDPM 里面没有被使用
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )
    def predict_eps_from_z_and_v(self, x_t, t, v):
        #？？？这tm 对吗？
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )
    #8-9 这两个是一组的，一起计算xt-1的概率公式的基于xt
    def q_posterior(self, x_start, x_t, t):
        #计算非常的标准
        #DDPM equation 7
        #q(xt-1|xt,x0) 但是是mean_t
        #mean variance 计算的非常准确
        #进一步去看看clip 的内容，
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        #截取到1e-20 再对上面的posterior_variance 取log计算，
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    def p_mean_variance(self, x, t, clip_denoised: bool):

        #p(x0|xt)  noise 也是xt预测的，感觉怎么有点像ddim 啊
        #！！！！注意，此处仍然不会是ddim,推导的基本元素都不一样像个毛
        # 此处就是简单的从xt 去获得x0 由于encoder 预测的是noise 那么
        # 使用predict_start_from_noise 的时候就把noise 替换一下
        # 最终是为了获得x0
        model_out = self.model(x, t)# from x_t to noise 
        #直接计算下面的东西
        if self.parameterization == "eps":
            #因为预测的是noise 那么x0就根据
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out

        if clip_denoised:x_recon.clamp_(-1., 1.)# 所有值卡在这个区间
        #里面的x0 是基于 x_t获得的 里面的计算的onise 也是基于x_t获得的
        #仍然#DDPM equation 7
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    ###################################
    #4个sample  3个采样过程就直接不要梯度了
    #10-13 先是p sample 三连 从噪声到原数据 再从x0 到x_t
    @torch.no_grad()#OK
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        #一个时刻的sample ！
        #prediction sample process
        #仍然DDPM e7 q(xt-1|xt,x0)
        # 基于x_t 得到mean 与logvar
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        # 最后一个时刻就不加噪声了，但是这个形式。。要不。。试试？
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        #为什么要这样采样？ 不能老老实实用var那么计算？你这样做有什么目的？
        #由于nonzero_mask的后三个维度都是所以通过广播可以自动扩展匹配张量x的规模。
        # 这意味着我的x的形式在这个位置可以是未知的，不需要在这个时候给定，我直接都有x shape了为什么不用呢？
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        #从噪声到x0的循环sample ！
        device = self.betas.device
        #shape 16 3 256 256 4个维度
        b = shape[0]
        img = torch.randn(shape, device=device)#注意，此时是噪声！
        intermediates = [img]
        #注意，这里将i 转化为t的操作 
        #构造了一个长度为b 的，值为i 的tensor 确实哦！ 对于batch 的每个instance 要单独给一个
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)

            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)

        if return_intermediates:return img, intermediates
        return img
    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        # 怎么在这个位置直接给出了batch size？这个可能需要改
        # 设定了大小
        # 输入就是个大小
        # 输出就是一张完整的image !
        image_size = self.image_size#256
        channels = self.channels#3
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)   
    def q_sample(self, x_start, t, noise=None):#这个是prior 吗 这么精确的
        # Get samples from the distribution q(x_t | x_0)
        # 这个是从前向后？
        # .q_mean_variance
        # 那么为啥不放上去
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


    #14-15
    def get_v(self, x, noise, t):
        #DDPM模型中用于生成图像的条件统计量v score
        #看unified 等价形式 基于x 和noise？
        #这个公式看起来仍然怪怪的
        #另外一个也有predict_v 
        #从x0出发计算的
        #质疑一下，虽然也没人用这个
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )
    def get_loss(self, pred, target, mean=True):
        #一个朴实无华的loss 的计算
        #基于predict and target 
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss
    #16-17
    def p_losses(self, x_start, t, noise=None):
        #基于Input x_start 得到x_t然后through model 得到预测的noise
        #并且是一个时刻的
 
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_out = self.model(x_noisy, t)#? e(x_t,t) 感觉这里是前向计算
        #这难道只有encoder了吗？
        loss_dict = {}
        if self.parameterization == "eps":
            target = noise#那必然是你了
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")
        #batch 的每一个都有一个自己的loss，你这是
        #x输入 预测噪声去了？
        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})#不是你这里又mean回去了？
        loss_simple = loss.mean() * self.l_simple_weight#weight=1

        loss_vlb = (self.lvlb_weights[t] * loss).mean()#反而是对初始时刻给了非常大的权重，
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb#0这个los

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict
    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        #我算是知道为啥不先均值batch 这个维度了 不同时刻的，但是最后还是要mean batch的
        return self.p_losses(x, t, *args, **kwargs)
    #18-20
    def get_input(self, batch, k):
        #我会有dataloader 来替换掉这个函数得到x
        x = batch[k]#image?
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')# 图像才用的我可能不用
        x = x.to(memory_format=torch.contiguous_format).float()
        return x
    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)#“就是个image” 啊得到x之后
        loss, loss_dict = self(x)#执行了forward 方法返回了2个Loss
        return loss, loss_dict
    def training_step(self, batch, batch_idx):
        #还是计算一下这个batch的loss的
        #并且每个人只计算了一个时间点的loss
        #这个空的啥也不执行
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val
        #拿了一组图片计算了一下loss
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
    #21-23
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        #应该是执行了对loss dict的存储相关的东西
        #不ema执行一个
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():#ema 执行了一个
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    def on_train_batch_end(self, *args, **kwargs):#没被执行
        if self.use_ema:#确实使用了这个
            self.model_ema(self.model)
    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)#
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid
    #24-25
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:#False了
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)# 基本山更久没有decay

       
        return opt

    def save_model(self, checkpoint_dir, iteration):
        # Save generators, discriminators, and optimizers
        Net_name = os.path.join(checkpoint_dir, 'DDPM'+str(iteration)+'.pt')        
        #opt_name = os.path.join(checkpoint_dir, 'optimizer.pt')       
        torch.save(self.state_dict(), Net_name)#这里面应该包含了吧 
    def resume(self, checkpoint_dir, iteration, test=False):
        # Load generators
        Net_name = os.path.join(checkpoint_dir, 'DDPM'+str(iteration)+'.pt')
        self.load_state_dict(torch.load(Net_name))
        iteration=0
        return iteration  


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.sequential_cross_attn = diff_model_config.pop("sequential_crossattn", False)
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm', 'hybrid-adm', 'crossattn-adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, c_adm=None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            out = self.diffusion_model(x, t, context=c_concat)
        elif self.conditioning_key == 'crossattn':
            if not self.sequential_cross_attn:
                #我懂了，他喵的这个是同时有多个conds 进行拼接，我目前就一个不需要啊
                cc = c_crossattn#torch.cat(c_crossattn, 1)
            else:
                cc = c_crossattn
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'hybrid-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm)
        elif self.conditioning_key == 'crossattn-adm':
            assert c_adm is not None
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, y=c_adm)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out



class Fin_Aug_diffusion(DDPM):
    """main class""" 
    #scale_factor: 0.18215  
    #cond_stage_key: "caption"
    #cond_stage_trainable: false
    #conditioning_key: crossattn
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,#0.18215 
                 scale_by_std=False,
                 force_null_conditioning=False,
                 *args, **kwargs):
        #3False 
        self.force_null_conditioning = force_null_conditioning
        self.num_timesteps_cond = default(num_timesteps_cond, 1)#1
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']#1<=1000
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:#crossattn
            conditioning_key = 'concat' if concat_mode else 'crossattn'

        if cond_stage_config == '__is_unconditional__' and not self.force_null_conditioning:
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)#
        reset_ema = kwargs.pop("reset_ema", False)#
        reset_num_ema_updates = kwargs.pop("reset_num_ema_updates", False)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)# 


        self.concat_mode = concat_mode#True 
        self.cond_stage_trainable = cond_stage_trainable#False，
        self.cond_stage_key = cond_stage_key #"caption" 
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1#2-1=1
        except:
            self.num_downs = 0

        if not scale_by_std:#False 
            self.scale_factor = scale_factor#0.18215 
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.cond_stage_forward = cond_stage_forward#False
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.restarted_from_ckpt = False


    def make_cond_schedule(self,):
        #获得一个[0,...,999]
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids#
   
    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        self.shorten_cond_schedule = self.num_timesteps_cond > 1#1>1 设置为了False.
        if self.shorten_cond_schedule:
            self.make_cond_schedule()




    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:#
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()#return self.mean
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    #14
    def forward(self, x, c, *args, **kwargs):

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:#latent diffustion 里面被设置成了crossattn 
            assert c is not None# 
            if self.cond_stage_trainable:#False  
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)
    #15 apply 5 papers,to 
 
    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]#
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}#"c_crossattn":[list]

        x_recon = self.model(x_noisy, t, **cond)#unet 的向前执行

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        #p(theta|x_t,x_0)
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]

        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        #Get the distribution q(x_t | x_0)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)


        #将单位从 nats （自然单位）转换成 bits （比特）
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)#x_t
        model_output = self.apply_model(x_noisy, t, cond)#unit 计算了

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])#
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        #################################
        #
#         print("start loss observation !")
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t#
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()
#         print('1 the loss is : {:.4f}'.format(loss.item()))
        #####################################################
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
#         print('2 with elbo the loss is : {:.4f}'.format(loss.item()))
        ##########################
        #2154 check status and add new term
#         abs_original =torch.mean(torch.abs(target)) 
#         abs_train =torch.mean(torch.abs(model_output))
#         abs_loss= torch.abs(abs_original-abs_train)  
#         loss += 0.3*abs_loss
#         loss_dict.update({f'{prefix}/loss': loss})

        x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=model_output)##Get the distribution p(x_0 
        abs_original = x_start.mean(dim=(1, 2, 3))
        abs_train    = x_recon.mean(dim=(1, 2, 3))
        abs_loss= torch.abs(abs_original-abs_train).mean()  
        loss += abs_loss
    
#         print('2 with abs the loss is : {:.4f}'.format(loss.item()))
#         y=asd+3
        loss_dict.update({f'{prefix}/loss': loss})        
        
        return loss, loss_dict
    #########################################################
    #19 +
    #q(xt-1|xt,x0)
    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        #首先获得了什么呢
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)
        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":

            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)##Get the distribution p(x_0 | x_t). 
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        #        #q(xt-1|xt,x0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance
    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs
        #基本上是前一刻的分布，然后获得一个Noise 了
        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise# 是标准差乘以noise 没有问题
    #21 denoising 好像也不是一个需要训练和梯度的过程
    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:#如果没有specified ，那么就使用默认的，
            log_every_t = self.log_every_t#设置成了100.每100次记一次
        timesteps = self.num_timesteps#1000
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:#这个是可能给的
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        #verbose 是true
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps#[11111]

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long) 
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)

                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            #q(xt-1|xt,x0) 无梯度的向前采样
            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t#100 
        device = self.betas.device#
        
        if isinstance(cond, list):
            b = cond[0].shape[0]
        else:
            b = cond.shape[0]
        c,h,w=shape[1:]
        shape_new=(b,c,h,w)
        if x_T is None:
            img = torch.randn(shape_new, device=device)# 
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:#
            timesteps = self.num_timesteps#

        if start_T is not None:
            timesteps = min(timesteps, start_T)


        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:# 
            ts = torch.full((b,), i, device=device, dtype=torch.long)#
            if self.shorten_cond_schedule:#
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img
    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:#batch_size 切片作用
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}

            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:#
            ddim_sampler = DDIMSampler(self)#这个self 里面包含了DDIM 所需要的参数了
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                         shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)

        return samples, intermediates
    @torch.no_grad()
    def ddim_sample_log(self, cond, data_shape, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)#这个self 里面包含了DDIM 所需要的参数了
        batch_size = data_shape[0]
        shape = (data_shape[1],data_shape[2],data_shape[3])   
        samples, _ = ddim_sampler.sample(ddim_steps, batch_size,shape, cond, verbose=False, **kwargs)
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt



class Fin_Aug_diffusion_csi(DDPM):
    """main class""" 
    #scale_factor: 0.18215  
    #cond_stage_key: "caption"
    #cond_stage_trainable: false
    #conditioning_key: crossattn
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,#0.18215 
                 scale_by_std=False,
                 force_null_conditioning=False,
                 *args, **kwargs):

        self.force_null_conditioning = force_null_conditioning
        self.num_timesteps_cond = default(num_timesteps_cond, 1)#
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']#1<=1000
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:#crossattn
            conditioning_key = 'concat' if concat_mode else 'crossattn'

        if cond_stage_config == '__is_unconditional__' and not self.force_null_conditioning:
            conditioning_key = None

        ckpt_path = kwargs.pop("ckpt_path", None)#
        reset_ema = kwargs.pop("reset_ema", False)#
        reset_num_ema_updates = kwargs.pop("reset_num_ema_updates", False)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)# 


        self.concat_mode = concat_mode#True 
        self.cond_stage_trainable = cond_stage_trainable#False，
        self.cond_stage_key = cond_stage_key #"caption" 
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1#2-1=1
        except:
            self.num_downs = 0#

        if not scale_by_std:#False 
            self.scale_factor = scale_factor#0.18215 
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.cond_stage_forward = cond_stage_forward#False
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.restarted_from_ckpt = False


    def make_cond_schedule(self,):
        #获得一个[0,...,999]
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids#第一个值为0
   
    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1#
        if self.shorten_cond_schedule:
            self.make_cond_schedule()




    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()#return self.mean
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    #14 正常的forward计算了
    def forward(self, x, c, *args, **kwargs):
        #还是正常时间，一个时间步，
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:#latent diffustion 里面被设置成了crossattn 
            assert c is not None# 
            if self.cond_stage_trainable:#False  
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)
    #15 apply 5 papers,to 
    # 可以叫recon through unet apply unet 吧，LDM
    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]#
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}#"c_crossattn":[list]

        x_recon = self.model(x_noisy, t, **cond)#u

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        #p(theta|x_t,x_0)

        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]

        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        #Get the distribution q(x_t | x_0)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)

        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)


        #将单位从 nats （自然单位）转换成 bits （比特）
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)#x_t
        model_output = self.apply_model(x_noisy, t, cond)#unit 计算

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])#
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        #################################
        #
#         print("start loss observation !")
        logvar_t = self.logvar[t].to(self.device)#
        loss = loss_simple / torch.exp(logvar_t) + logvar_t#
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()
#         print('1 the loss is : {:.4f}'.format(loss.item()))
        #####################################################

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})        
        
        return loss, loss_dict
    #########################################################
    #19 +
    #q(xt-1|xt,x0)
    #只有p sample 里面使用了，
    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t

        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)
        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":

            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)##Get the distribution p(x_0 | x_t).
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        #        #q(xt-1|xt,x0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance
    #20

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise# 
 
    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:#，
            log_every_t = self.log_every_t#
        timesteps = self.num_timesteps#1000
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:#
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:#
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        #verbose 是true
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps#[11111]

        for i in iterator:#
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)#
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)

                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            #q(xt-1|xt,x0) 无梯度的向前采样
            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t#
        device = self.betas.device#
        
        if isinstance(cond, list):
            b = cond[0].shape[0]
        else:
            b = cond.shape[0]
        c,h,w=shape[1:]
        shape_new=(b,c,h,w)
        if x_T is None:
            img = torch.randn(shape_new, device=device)# 
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps#

        if start_T is not None:
            timesteps = min(timesteps, start_T)


        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  

        for i in iterator:# 
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:#batch_size 切片作用
            if isinstance(cond, dict):

                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}

            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)#
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                         shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)

        return samples, intermediates

    @torch.no_grad()
    def ddim_sample_log(self, cond, data_shape, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)#
        batch_size = data_shape[0]
        shape = (data_shape[1],data_shape[2],data_shape[3])   
        samples, _ = ddim_sampler.sample(ddim_steps, batch_size,shape, cond, verbose=False, **kwargs)
        return samples

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt
