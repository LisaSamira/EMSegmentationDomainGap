"""Epochen-basiertes Training + Evaluation (Foreground-IoU, Dice) mit Zeitmessung und W&B-Logging.

Einzelner Lauf:
    python train.py --data_root /pfad/cem_mitolab --datasets "Bleck2018_SoleusBV2" --train_size 25 --wandb_mode offline
Alle Keys aus DEFAULTS sind als --key Wert überschreibbar (auch durch einen wandb-Sweep).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from dataset import get_dataloaders
from model import build_model

DEFAULTS = dict(
    # --- Daten ---
    data_root="data/EMPIAR-11037/data/cem_mitolab",
    datasets="Bleck2018_SoleusBV2",  # genau ein Sub-Datensatz (Datensätze für saubere Domänen nicht mischen!)
    train_size=25,             # 15 | 25 | 50 | 100 | 200 | 400
    val_size=25,               # fest, unabhängig von train_size
    test_size=25,              # fest, unabhängig von train_size
    split_mode="contiguous",   # Standard: "contiguous" verhindert Z-Slice-Leakage bei 3D-Volumina; sonst "random"
    split_seed=0,
    img_size=448,              # Vielfaches von 16 (Patchgröße DINOv3)
    num_workers=4,
    cache_data=True,
    # --- Augmentation ---
    aug_flip=True,
    aug_rot90=True,
    aug_rot_deg=0.0,           # >0 aktiviert freie Rotation (±deg)
    aug_intensity=0.2,
    # --- Modell ---
    model="dinov3",
    backbone_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
    n_feature_layers=1,        # 1 = nur letzte Schicht, 4 = 4 Schichten konkateniert
    head_channels=256,
    head_dropout=0.1,
    # --- Optimierung ---
    epochs=100,
    batch_size=8,
    lr=1e-3,
    weight_decay=1e-2,
    warmup_frac=0.05,
    grad_clip=1.0,
    dice_weight=1.0,
    threshold=0.5,
    patience=20,               # Early Stopping auf val IoU; 0 = aus
    amp=True,
    seed=0,
    # --- Logging ---
    wandb_project="sdg-datasize-tests",
    wandb_entity=None,
    wandb_mode="online",       # online | offline (only saves logs locally in wandb/ folder) | disabled
    out_dir="runs",
)


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DATASET_TAGS = {
    "Bleck2018_SoleusBV2": "Bleck2018",
    "Wei2020_MitoEM-H": "Wei2020H",
    "Wei2020_MitoEM-R": "Wei2020R",
    "314W440Y68Pf573BO3p2_3D": "314W440",
    "5j54Gst6j5069xCUpF5M_2D": "5j54Gst",
}


def dataset_tag(datasets: str | list[str] | None) -> str:
    """Kurzer Datensatzname für Ordnernamen, z. B. 'Bleck2018_SoleusBV2' -> 'Bleck2018'."""
    if not datasets:
        return "all"
    if isinstance(datasets, str):
        datasets = datasets.split(",")
    first = str(datasets[0]).strip()
    if first in DATASET_TAGS:
        return DATASET_TAGS[first]
    tag = first.split("_")[0]
    tag = re.sub(r"[^A-Za-z0-9.-]+", "", tag)
    return tag or "dataset"


class Timer:
    """Wall-clock-Timer mit CUDA-Synchronisation (sonst wären GPU-Zeiten falsch)."""

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        self.elapsed = 0.0

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0


class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, eps: float = 1.0):
        super().__init__()
        self.w, self.eps = dice_weight, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target)
        if self.w == 0:
            return bce
        p = torch.sigmoid(logits)
        inter = (p * target).sum((1, 2, 3))
        dice = (2 * inter + self.eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + self.eps)
        return bce + self.w * (1 - dice).mean()


def make_scheduler(opt, total_steps: int, warmup_steps: int):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


@torch.no_grad()
def evaluate(model, loader, device, criterion, threshold=0.5, amp=True) -> dict:
    """Foreground-IoU & Dice. Primär: Mikro-Mittel über alle Pixel des Datensatzes;
    zusätzlich: Mittel über Einzelbilder (leere GT+Pred zählt als 1.0). Wiederverwendbar für Cross-Dataset-Tests."""
    model.eval()
    tp = fp = fn = 0
    loss_sum, n = 0.0, 0
    per_iou, per_dice = [], []
    with Timer(device) as t:
        for b in loader:
            x = b["image"].to(device, non_blocking=True)
            y = b["mask"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                logits = model(x)
            logits = logits.float()
            loss_sum += criterion(logits, y).item() * x.size(0)
            n += x.size(0)
            pred, tgt = logits.sigmoid() > threshold, y > 0.5
            d = (1, 2, 3)
            tp_i, fp_i, fn_i = (pred & tgt).sum(d), (pred & ~tgt).sum(d), (~pred & tgt).sum(d)
            tp += tp_i.sum().item(); fp += fp_i.sum().item(); fn += fn_i.sum().item()
            union = tp_i + fp_i + fn_i
            den = 2 * tp_i + fp_i + fn_i
            per_iou.append(torch.where(union == 0, torch.ones_like(union, dtype=torch.float),
                                       tp_i / union.clamp(min=1)).cpu())
            per_dice.append(torch.where(den == 0, torch.ones_like(den, dtype=torch.float),
                                        2 * tp_i / den.clamp(min=1)).cpu())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0
    return dict(iou=iou, dice=dice, iou_img=torch.cat(per_iou).mean().item(),
                dice_img=torch.cat(per_dice).mean().item(), loss=loss_sum / max(n, 1),
                time_s=t.elapsed, n_images=n, ms_per_image=1000 * t.elapsed / max(n, 1))


# --------------------------------------------------------------------------- #
# Hauptfunktion
# --------------------------------------------------------------------------- #
def run(overrides: dict | None = None) -> dict:
    t_run0 = time.perf_counter()
    cfg = {**DEFAULTS, **(overrides or {})}
    wb = wandb.init(project=cfg["wandb_project"], entity=cfg.get("wandb_entity"), mode=cfg["wandb_mode"],
                    config=cfg, group=f"train_size_{cfg['train_size']}", tags=[f"n{cfg['train_size']}", cfg["model"]])
    cfg = {**DEFAULTS, **dict(wandb.config)}  # Sweep-Werte haben Vorrang
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg["out_dir"]) / f"{dataset_tag(cfg['datasets'])}_n{cfg['train_size']}_{wb.id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with Timer(device) as t_data:
        loaders, info = get_dataloaders(cfg)
    with open(run_dir / "split.json", "w") as f:
        json.dump(info, f, indent=2)
    with Timer(device) as t_model:
        model = build_model(cfg).to(device)
    n_train_p = sum(p.numel() for p in model.trainable_parameters())
    n_total_p = sum(p.numel() for p in model.parameters())
    wandb.config.update({"trainable_params": n_train_p, "total_params": n_total_p, "n_images": info["n"],
                         "val_files": info["val_files"], "test_files": info["test_files"],
                         "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
                        allow_val_change=True)
    wandb.log({"time/setup_data_s": t_data.elapsed, "time/setup_model_s": t_model.elapsed}, step=0)
    print(f"[run] n_train={info['n']['train']} val={info['n']['val']} test={info['n']['test']} "
          f"| trainable {n_train_p/1e6:.2f}M / total {n_total_p/1e6:.1f}M")

    criterion = BCEDiceLoss(cfg["dice_weight"])
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * cfg["epochs"]
    sched = make_scheduler(opt, total_steps, int(cfg["warmup_frac"] * total_steps))
    use_amp = bool(cfg["amp"]) and device.type == "cuda"

    best = dict(iou=-1.0, epoch=0)
    best_path = run_dir / "best_head.pt"
    train_time_total = 0.0
    bad_epochs = 0
    epoch = 0
    test = None
    status = "completed"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            loss_sum = torch.zeros((), device=device)
            seen, wait = 0, 0.0
            with Timer(device) as t_ep:
                t_prev = time.perf_counter()
                for b in loaders["train"]:
                    wait += time.perf_counter() - t_prev  # Zeit, die auf den DataLoader gewartet wurde
                    x = b["image"].to(device, non_blocking=True)
                    y = b["mask"].to(device, non_blocking=True)
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                        logits = model(x)
                    loss = criterion(logits.float(), y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    if cfg["grad_clip"] > 0:
                        nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg["grad_clip"])
                    opt.step()
                    sched.step()
                    loss_sum += loss.detach() * x.size(0)
                    seen += x.size(0)
                    t_prev = time.perf_counter()
            train_time_total += t_ep.elapsed

            val = evaluate(model, loaders["val"], device, criterion, cfg["threshold"], use_amp)
            improved = val["iou"] > best["iou"]
            if improved:
                best = dict(iou=val["iou"], dice=val["dice"], epoch=epoch)
                torch.save(model.trainable_state_dict(), best_path)
                bad_epochs = 0
            else:
                bad_epochs += 1
            wandb.log({
                "epoch": epoch, "lr": sched.get_last_lr()[0],
                "train/loss": (loss_sum / seen).item(),
                "val/loss": val["loss"], "val/iou": val["iou"], "val/dice": val["dice"],
                "val/iou_img": val["iou_img"], "val/dice_img": val["dice_img"],
                "val/best_iou": best["iou"],  # monoton -> geeignete Sweep-Metrik
                "time/train_epoch_s": t_ep.elapsed, "time/train_dataload_wait_s": wait,
                "time/val_s": val["time_s"], "time/train_total_s": train_time_total,
            }, step=epoch)
            if cfg["patience"] > 0 and bad_epochs >= cfg["patience"]:
                print(f"[run] Early Stopping nach Epoche {epoch} (bester: {best['epoch']})")
                break

        # --- Test mit bestem Checkpoint (Auswahl nur über Val!) ---
        model.head.load_state_dict(torch.load(best_path, map_location=device))
        test = evaluate(model, loaders["test"], device, criterion, cfg["threshold"], use_amp)
    except BaseException:
        status = "interrupted"
        raise
    finally:
        # Test auch nach einem Abbruch mit dem besten Checkpoint nachholen.
        if test is None and best["epoch"] > 0 and best_path.exists():
            try:
                model.head.load_state_dict(torch.load(best_path, map_location=device))
                test = evaluate(model, loaders["test"], device, criterion, cfg["threshold"], use_amp)
            except Exception as e:
                print(f"[run] Test-Evaluation nach Abbruch fehlgeschlagen: {e}")
        peak_mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        total_time = time.perf_counter() - t_run0
        summary = {
            "best_epoch": best["epoch"], "best_val_iou": best["iou"], "epochs_run": epoch,
            "time/train_total_s": train_time_total,
            "time/train_mean_epoch_s": train_time_total / max(epoch, 1),
            "time/run_total_s": total_time, "gpu_peak_mem_gb": peak_mem,
            "status": status,
        }
        if test is not None:
            summary.update({
                "test/iou": test["iou"], "test/dice": test["dice"], "test/iou_img": test["iou_img"],
                "test/dice_img": test["dice_img"], "test/loss": test["loss"],
                "time/test_s": test["time_s"], "time/test_ms_per_image": test["ms_per_image"],
            })
        try:
            if wandb.run is not None:
                for k, v in summary.items():
                    wandb.run.summary[k] = v
        except Exception:
            pass
        with open(run_dir / "results.json", "w") as f:
            json.dump({"config": cfg, "split": info, "best": best, "val_at_best": best,
                       "test": test, "summary": summary}, f, indent=2, default=str)
        if test is not None:
            print(f"[run] val IoU {best['iou']:.4f} (Ep {best['epoch']}) | test IoU {test['iou']:.4f} "
                  f"Dice {test['dice']:.4f} | train {train_time_total:.1f}s test {test['time_s']:.2f}s")
        else:
            print(f"[run] {status}: val IoU {best['iou']:.4f} (Ep {best['epoch']}) nach {epoch} Epochen")
        try:
            wandb.finish()
        except Exception:
            pass
    return summary


# --------------------------------------------------------------------------- #
# CLI (alle DEFAULTS-Keys automatisch als Argumente; kompatibel mit `wandb agent`)
# --------------------------------------------------------------------------- #
def _str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "t", "yes", "y")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", type=_str2bool, default=v)
        elif v is None:
            p.add_argument(f"--{k}", type=str, default=None)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    return p


if __name__ == "__main__":
    run(vars(build_parser().parse_args()))
