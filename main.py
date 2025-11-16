import os
import time
import math
import yaml
import numpy as np

from utils import dataset as dts
from utils import models
from utils.models import FNO1d_Bayes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch

# ===== Ngưỡng coverage tối thiểu =====
COV_MIN = 93.0
COV_TARGET = 95.0
COV_TOL_TIE = 1e-6  # chỉ để ràng buộc ổn định khi so PIw rất sát

# ===== Khóa so sánh: Ưu tiên PIw nhỏ nhất, chỉ xét nếu cov95 >= 94% =====
def make_best_key(cov95_mean, width_mean, cov_min=COV_MIN, cov_target=COV_TARGET):
    """
    Trả về tuple dùng cho min():
    - Nếu cov95 < cov_min -> trả về (inf, inf) để không bao giờ thắng.
    - Ngược lại -> (width_mean, |cov-95|)  (PIw nhỏ nhất trước, rồi gần 95% hơn)
    """

    if cov95_mean < cov_min:
        return (float('inf'), float('inf'))
    cov_gap = abs(cov95_mean - cov_target)
    return (width_mean, cov_gap)

def train(model, model_config):

    #init optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    num_epochs = model_config["epoch"]
    num_batches = len(train_loader)
    # ===== Optional: scheduler =====
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.8, patience=50)
    # ===== Training loop hoàn chỉnh =====
    save_path = model_config["save_path"]
    best_val_metric = float('inf')
    patience, bad_epochs = 50, 0
    best_key = None  # lưu tuple (pass_flag, cov_gap, width)
    best_epoch = None


    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_nll = 0.0
        running_kl = 0.0

        # KL annealing: tăng beta 0->1 trong 20 epoch đầu
        beta = 1e-6 * min(1.0, epoch / 20.0)

        for xb, yb in train_loader:  # xb:[B,C_in,N], yb:[B,C_out,N]
            xb, yb = xb.to(device), yb.to(device)
            mu, logvar, kl = model(xb)
            loss, logs = FNO1d_Bayes.elbo_gaussian_nll(mu, logvar, yb, kl,
                                           batch_size=xb.size(0),
                                           num_batches=num_batches,
                                           beta=beta)
            mse_loss = torch.nn.functional.mse_loss(mu, yb)
            loss += 1e3 * mse_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            running_loss += loss.item()
            running_nll += logs['nll'].item()
            running_kl += logs['kl_scaled'].item()

        train_loss = running_loss / len(train_loader)
        train_nll = running_nll / len(train_loader)
        train_kl = running_kl / len(train_loader)

        # ===== Validation + UQ qua MC =====
        model.eval()
        with torch.no_grad():
            if 'val_loader' in globals() and val_loader is not None and len(val_loader) > 0:
                val_loss = 0.0
                val_nll = 0.0
                val_kl = 0.0
                cov95_list, width_list, mse_list = [], [], []

                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    # ELBO với beta=1 khi eval
                    mu, logvar, kl = model(xb)
                    loss, logs = FNO1d_Bayes.elbo_gaussian_nll(mu, logvar, yb, kl,
                                                   batch_size=xb.size(0),
                                                   num_batches=1, beta=beta)
                    val_loss += loss.item()
                    val_nll += logs['nll'].item()
                    val_kl += logs['kl_scaled'].item()

                    # MC để tách aleatoric/epistemic + coverage
                    pm, ale, epi, tot = FNO1d_Bayes.mc_predict(model, xb, T=32)
                    cov95, width, mse = models.pi95_stats(yb, pm, tot)
                    cov95_list.append(cov95);
                    width_list.append(width);
                    mse_list.append(mse)
                ale_mean = torch.mean(ale).item()
                epi_mean = torch.mean(epi).item()
                print(f"Aleatoric mean var={ale_mean:.6e} | Epistemic mean var={epi_mean:.6e}")
                val_loss /= len(val_loader)
                val_nll /= len(val_loader)
                val_kl /= len(val_loader)
                cov95_mean = sum(cov95_list) / len(cov95_list)
                width_mean = sum(width_list) / len(width_list)
                mse_mean = sum(mse_list) / len(mse_list)

                # Scheduler dựa trên NLL validation (ổn định hơn)
                scheduler.step(val_nll)

                # ===== SAVE BEST theo tiêu chí (cov95 ~ 95% & PIw nhỏ nhất) =====
                current_key = make_best_key(cov95_mean, width_mean)

                if (best_key is None) or (current_key < best_key):
                    if math.isfinite(current_key[0]):  # chỉ lưu nếu đạt ngưỡng cov95 >= 94%
                        best_key = current_key
                        best_epoch = epoch
                        bad_epochs = 0
                        torch.save({'epoch': epoch,
                                    'model_state': model.state_dict(),
                                    'opt_state': opt.state_dict(),
                                    'best_key': best_key,
                                    'cov95_mean': cov95_mean,
                                    'width_mean': width_mean,
                                    'mse_mean': mse_mean}, save_path)
                        best_tag = " (saved)"
                    else:
                        best_tag = ""  # không lưu vì chưa đạt ngưỡng
                else:
                    bad_epochs += 1
                    best_tag = ""

                print(f"[Epoch {epoch:03d}] "
                      f"Train: loss={train_loss:.4f} | nll={train_nll:.4f} | klβ={train_kl:.4f} | MSE={mse_loss:.4f} || "
                      f"Val: loss={val_loss:.4f} | nll={val_nll:.4f} | klβ={val_kl:.4f} | "
                      f"cov95={cov95_mean:.2f}% | PIw={width_mean:.6f} | MSE={mse_mean:.6f} "
                      f"| key(width,|cov-95|)={tuple(round(x, 6) if math.isfinite(x) else x for x in current_key)}{best_tag}")

                if bad_epochs >= patience:
                    print(
                        f"> Early stop at epoch {epoch} (no improve {patience} epochs). Best at epoch {best_epoch} with key={best_key}")
                    break
            else:
                # Nếu không có val_loader, vẫn in train logs và chạy quick MC trên một batch train
                xb, yb = next(iter(train_loader))
                xb, yb = xb.to(device), yb.to(device)
                pm, ale, epi, tot = FNO1d_Bayes.mc_predict(model, xb[:min(8, xb.size(0))], T=32)
                cov95, width, mse = FNO1d_Bayes.pi95_stats(yb[:pm.size(0)], pm, tot)
                print(f"[Epoch {epoch:03d}] "
                      f"Train: loss={train_loss:.4f} | nll={train_nll:.4f} | klβ={train_kl:.4f} | β={beta:.2f} || "
                      f"QuickMC: cov95={cov95:.1f}% | PIw={width:.6f} | MSE={mse:.6f}")

    # ===== Load best (nếu có validation) =====

    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        print(f"Loaded best checkpoint from epoch {ckpt.get('epoch', -1)} "
              f"with cov95={ckpt.get('cov95_mean', float('nan')):.2f}%, "
              f"PIw={ckpt.get('width_mean', float('nan')):.6f}, MSE={ckpt.get('mse_mean', float('nan')):.6f}, "
              f"best_key={ckpt.get('best_key', None)}")
        t_start = time.time()
        model.eval()
        all_pred_means = []
        all_ale_vars = []
        all_epi_vars = []
        all_total_vars = []
        all_targets_val = []  # To store actual y_test values

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pm, ale, epi, tot = FNO1d_Bayes.mc_predict(model, xb, T=16)  # Using 16 MC samples

                all_pred_means.append(pm.cpu())
                all_ale_vars.append(ale.cpu())
                all_epi_vars.append(epi.cpu())
                all_total_vars.append(tot.cpu())
                all_targets_val.append(yb.cpu())

        # Concatenate results from all batches
        val_pred_mean = torch.cat(all_pred_means, dim=0)
        val_ale_var = torch.cat(all_ale_vars, dim=0)
        val_epi_var = torch.cat(all_epi_vars, dim=0)
        val_total_var = torch.cat(all_total_vars, dim=0)
        val_targets = torch.cat(all_targets_val, dim=0)


        print(f"Validation Predicted Mean (destandardized) shape: {val_pred_mean.shape}")
        print(f"Validation Total Variance (destandardized) shape: {val_total_var.shape}")
        print(f"Validation Targets (destandardized) shape: {val_targets.shape}")

        # Flatten the destandardized predictions and targets for metric calculation
        preds_flat = val_pred_mean.cpu().numpy().flatten()
        targets_flat = val_targets.cpu().numpy().flatten()

        # Calculate metrics
        mse = mean_squared_error(targets_flat, preds_flat)
        mae = mean_absolute_error(targets_flat, preds_flat)
        rmse = np.sqrt(mse)  # RMSE is the square root of MSE
        r2 = r2_score(targets_flat, preds_flat)

        t_end = time.time()

        print(f"Metric calculation time: {t_end - t_start:.4f} seconds")
        print("\nValidation Metrics:")
        print(f"Mean Squared Error (MSE): {mse:.8f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.8f}")
        print(f"Mean Absolute Error (MAE): {mae:.8f}")
        print(f"R-squared (R2): {r2:.8f}")
    else:
        print("Chưa có epoch nào đạt cov95 ≥ 94%, nên chưa lưu checkpoint.")

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Warning: Model will be trained in {device}')
    print("*"*50)
    with open("configs/config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    data_config = config["data"]
    model_config = config["model"]

    INPUT = model_config["input"]
    OUTPUT = model_config["output"]
    MODES = model_config["modes"]
    WIDTH = model_config["width"]
    PRIOR_SIGMA = model_config["prior_sigma"]
    FINETUNING = model_config["finetuning"]
    freeze = model_config["freeze"]

    #Load dataset
    train_loader, val_loader = dts.load_dataset(data_config)

    #init BNNFNO model

    model = FNO1d_Bayes(in_channels=INPUT, out_channels=OUTPUT, modes=MODES, width=WIDTH, prior_sigma=PRIOR_SIGMA).to(device)
    if FINETUNING:
        model = models.load_fno_blocks_into_bnn(model, model_config["pretrained"], freeze = freeze)
    #=======Model trainning!=====
    train(model, model_config)