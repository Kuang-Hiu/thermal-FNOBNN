import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import re
from copy import deepcopy
# ========= Spectral block  =========
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.weights = nn.Parameter(torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat) * 0.1)

    def compl_mul1d(self, input, weights):
        return torch.einsum("bcn,con->bon", input, weights)

    def forward(self, x):
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.out_channels, x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)
        x = torch.fft.irfft(out_ft, n=N, dim=-1)
        return x

# ========= Bayesian 1x1 Conv (variational) =========
class BayesianConv1d(nn.Module):
    """
    1x1 conv variational:
      out[b, o, n] = sum_c W[o,c] * x[b,c,n] + b[o]
    W ~ N(mu, sigma^2), sigma = softplus(rho).
    Return: out, kl_div (prior N(0,1)).
    """
    def __init__(self, in_channels, out_channels, bias=True, prior_mu=0.0, prior_sigma=1e-1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        # Posterior q(W): dùng ma trận [out,in] (không có trục 1 chiều)
        self.weight_mu  = nn.Parameter(torch.empty(out_channels, in_channels))
        self.weight_rho = nn.Parameter(torch.empty(out_channels, in_channels))
        if bias:
            self.bias_mu  = nn.Parameter(torch.empty(out_channels))
            self.bias_rho = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -4.0)  # sigma init
        if self.bias_mu is not None:
            bound = 1. / max(1, self.in_channels) ** 0.5
            nn.init.uniform_(self.bias_mu, -bound, bound)
            nn.init.constant_(self.bias_rho, -4.0)

    @staticmethod
    def _softplus(x):
        return F.softplus(x)

    def kl_divergence(self, w_mu, w_sigma, prior_mu, prior_sigma):

        # KL(q||p) each Gaussian
        term1 = torch.log(prior_sigma / (w_sigma + 1e-12))
        term2 = (w_sigma**2 + (w_mu - prior_mu )**2) / (2 * (prior_sigma**2))
        kl = (term1 + term2 - 0.5).sum()
        return kl

    def forward(self, x):
        """
        x: [B, C_in, N]
        out: [B, C_out, N]
        """
        B, C, N = x.shape
        device = x.device

        # Sample W, b
        w_sigma = self._softplus(self.weight_rho)
        W = self.weight_mu + w_sigma * torch.randn_like(self.weight_mu, device=device)

        if self.bias_mu is not None:
            b_sigma = self._softplus(self.bias_rho)
            b = self.bias_mu + b_sigma * torch.randn_like(self.bias_mu, device=device)
        else:
            b = None

        # 1x1 conv theo kênh: giữ nguyên N
        # (broadcast b theo N)
        out = torch.einsum('bcn,oc->bon', x, W)
        if b is not None:
            out = out + b.view(1, -1, 1)

        # KL cho W và b
        kl = self.kl_divergence(self.weight_mu, w_sigma,
                                torch.tensor(self.prior_mu, device=device),
                                torch.tensor(self.prior_sigma, device=device))
        if self.bias_mu is not None:
            kl = kl + self.kl_divergence(self.bias_mu, self._softplus(self.bias_rho),
                                         torch.tensor(self.prior_mu, device=device),
                                         torch.tensor(self.prior_sigma, device=device))
        return out, kl
# ========= FNO với Bayesian head ouput (mu, logvar) =========
class FNO1d_Bayes(nn.Module):
    def __init__(self, in_channels, out_channels, modes=12, width=64, prior_sigma=1.0):
        super().__init__()
        self.out_channels = out_channels
        self.fc0 = nn.Conv1d(in_channels, width, 1)  # [B, C_in, N]
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.fc1 = nn.Conv1d(width, 128, 1)

        # Bayesian head: output = 2*C_out = mu và logvar
        self.fc2_bayes = BayesianConv1d(128, 2* out_channels, prior_sigma=prior_sigma)

    def forward(self, x):
        """
        Input:  x [B, C_in, N]
        Output: mu [B, C_out, N], logvar [B, C_out, N], kl (scalar tensor)
        """
        x = self.fc0(x)
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x = self.fc1(x)
        x = F.gelu(x)

        out, kl = self.fc2_bayes(x)  # [B, 2*C_out, N]
        mu, logvar = torch.split(out, self.out_channels, dim=1)
        # Stabilize logvar (optional clamp)
        logvar = torch.clamp(logvar, min=-15.0, max=-10)
        return mu, logvar, kl

    # ========= ELBO loss =========
    def elbo_gaussian_nll(mu, logvar, y, kl, batch_size, num_batches, beta=None):
        """
        Negative log likelihood (heteroscedastic) + KL.
        NLL per point: 0.5*(log(2π) + logσ² + (y-μ)²/σ²)
        KL per each batch: expected KL in epoch.
        beta:regular coef.  KL annealing; if None, use 1/num_batches (free bits).
        """
        if beta is None:
            beta = 1.0 / num_batches

        var = torch.exp(logvar) + 1e-12
        nll = 0.5 * (math.log(2 * math.pi) + logvar + (y - mu) ** 2 / var)
        nll = nll.mean()  # trung bình toàn batch/canal/pos

        elbo = nll + beta * (kl / batch_size)
        return elbo, {'nll': nll.detach(), 'kl_scaled': (beta * kl / batch_size).detach()}

    # ========= Predictive distribution with MC sampling =========
    @torch.no_grad()
    def mc_predict(model, x, T=20):
        """
        Inference T sample ( each sampe is drawned from p(w|D)).
        Return:
          pred_mean  [B,C,N]        = E_q[μ]
          ale_var    [B,C,N]        = E_q[σ²]
          epi_var    [B,C,N]        = Var_q[μ]
          total_var  [B,C,N]        = ale_var + epi_var
        """
        mus = []
        vars_ = []
        for _ in range(T):
            mu, logvar, _ = model(x)
            mus.append(mu)
            vars_.append(torch.exp(logvar))
        mus = torch.stack(mus, dim=0)  # [T,B,C,N]
        vars_ = torch.stack(vars_, dim=0)

        pred_mean = mus.mean(0)
        ale_var = vars_.mean(0)
        epi_var = mus.var(0, unbiased=False)
        total_var = ale_var + epi_var
        return pred_mean, ale_var, epi_var, total_var

def load_fno_blocks_into_bnn(model, ckpt_path, freeze = True):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # Lấy state_dict phù hợp
    sd = ckpt.get('state_dict', ckpt.get('model', ckpt))
    # Nếu có tiền tố 'module.' (DataParallel) thì bỏ đi
    sd = {re.sub(r'^module\.', '', k): v for k, v in sd.items()}

    own = model.state_dict()
    copied = []

    # Các lớp FNO cần copy (đến fc1, không copy head Bayesian)
    fno_prefixes = ('fc0.', 'conv1.', 'conv2.', 'w1.', 'w2.', 'fc1.')

    for k, v in sd.items():
        if k.startswith(fno_prefixes) and k in own and own[k].shape == v.shape:
            own[k] = v
            copied.append(k)

    # Nạp lại state_dict (bỏ qua các khóa không khớp)
    model.load_state_dict(own, strict=False)

    # frezzen FNO (fc0, conv1/2, w1/2, fc1)
    if freeze:
        for name, p in model.named_parameters():
            if name.startswith(('fc0', 'conv1', 'conv2', 'w1', 'w2', 'fc1')):
                p.requires_grad = False
            else:
                p.requires_grad = True  # fc2_bayes.* được mở khóa
    return model

# ===== Helper: PI95 coverage & width =====
@torch.no_grad()
def pi95_stats(y_true, pred_mean, total_var):
    """
    y_true, pred_mean, total_var: [B,C,N]
    Trả về: coverage (%), mean_width (trung bình độ rộng PI), mse
    """
    std = torch.sqrt(total_var.clamp_min(1e-12))
    lo = pred_mean - 1.96 * std
    hi = pred_mean + 1.96 * std
    inside = (y_true >= lo) & (y_true <= hi)
    coverage = inside.float().mean().item() * 100.0
    mean_width = (hi - lo).mean().item()
    mse = ((y_true - pred_mean)**2).mean().item()
    return coverage, mean_width, mse