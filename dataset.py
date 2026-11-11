"""Daten-Pipeline für binäre Mitochondrien-Segmentierung (2D) auf CEM-MitoLab-artigen Ordnern.

Erwartete Struktur (EMPIAR-11037, bereits in 2D-TIFFs konvertiert):

    <data_root>/                      # z. B. .../data/cem_mitolab
        <sub_dataset>/images/<name>.tiff
        <sub_dataset>/masks/<name>.tiff      # gleicher Dateiname wie das Bild

Nützlich zum Auswählen von Sub-Datensätzen:
    python dataset.py --root /pfad/zu/cem_mitolab --min_images 100
"""
from __future__ import annotations

import argparse
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TIFF_EXTS = (".tif", ".tiff")


# --------------------------------------------------------------------------- #
# Dateien finden
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Pair:
    image: Path
    mask: Path
    dataset: str

    @property
    def name(self) -> str:
        return f"{self.dataset}/{self.image.name}"


def natural_sort_key(s: str) -> tuple[list[tuple[int, int | str]], str]:
    """Zerlegt einen String für 'Natural Sorting' (z. B. heart_image2 vor heart_image10).

    Verhindert Typ-Konflikte (str vs. int in Python 3) durch getypte 2-Tupel:
    (0, str_lower) für Text, (1, int_val) für Zahlenblöcke.
    Garantiert deterministisches, stabiles Sortieren über heterogene Benennungsschemata
    (LOC-Tags, ROI-Tags, unpadded Slice-Nummern, Hashes).
    """
    tokens = re.split(r"(\d+)", s)
    key: list[tuple[int, int | str]] = []
    for t in tokens:
        if not t:
            continue
        if t.isdigit():
            key.append((1, int(t)))
        else:
            key.append((0, t.lower()))
    return (key, s)


def _list_tiffs(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in TIFF_EXTS),
        key=lambda p: natural_sort_key(p.name),
    )


def list_datasets(root: str | Path) -> dict[str, int]:
    """Alle Sub-Datensätze (Ordner mit images/ und masks/) samt Bildanzahl."""
    root = Path(root)
    out = {}
    for d in sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: natural_sort_key(p.name),
    ):
        if (d / "images").is_dir() and (d / "masks").is_dir():
            out[d.name] = len(_list_tiffs(d / "images"))
    return out


def parse_dataset_arg(datasets: str | list[str] | None) -> list[str] | None:
    """'all' / None -> None (= alle). Komma-getrennter String oder Liste -> Liste."""
    if datasets is None:
        return None
    if isinstance(datasets, str):
        datasets = [d.strip() for d in datasets.split(",") if d.strip()]
    if not datasets or [d.lower() for d in datasets] == ["all"]:
        return None
    return list(datasets)


def discover_pairs(root: str | Path, datasets: str | list[str] | None = None) -> list[Pair]:
    root = Path(root)
    names = parse_dataset_arg(datasets) or list(list_datasets(root).keys())
    pairs: list[Pair] = []
    for ds in names:
        img_dir, mask_dir = root / ds / "images", root / ds / "masks"
        if not img_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"{ds}: images/ oder masks/ fehlt unter {root / ds}")
        missing = 0
        for img in _list_tiffs(img_dir):
            mask = mask_dir / img.name
            if mask.exists():
                pairs.append(Pair(img, mask, ds))
            else:
                missing += 1
        if missing:
            warnings.warn(f"{ds}: {missing} Bilder ohne passende Maske übersprungen")
    if not pairs:
        raise RuntimeError(f"Keine Bild/Masken-Paare unter {root} gefunden.")
    return pairs


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def make_splits(
    pairs: list[Pair],
    n_train: int,
    n_val: int,
    n_test: int,
    split_seed: int = 0,
    mode: str = "contiguous",
) -> dict[str, list[Pair]]:
    """Feste Val-/Test-Mengen + geschachtelte Trainingsmengen.

    Val und Test hängen NICHT von n_train ab. Der Trainings-Pool wird mit split_seed gemischt,
    train = pool[:n_train]. Dadurch gilt train(15) ⊂ train(25) ⊂ ... ⊂ train(400): Unterschiede
    zwischen den Größen sind echte Größeneffekte und keine Stichproben-Effekte.

    mode="contiguous" (Standard): pro Sub-Datensatz werden Val/Test als zusammenhängende Blöcke
                       vom Ende der sortierten Liste genommen. Achtung: gruppiert nur nach
                       Sub-Datensatz, nicht nach LOC/Volume -> bei mehreren LOCs kann der
                       Schnitt durch ein Volume laufen (Z-Slice-Leakage zwischen Train und Val).
    mode="random":     globales Mischen (Gefahr: Nachbar-Slices desselben Volumens in Train & Test).
    """
    pairs = sorted(
        pairs,
        key=lambda p: (natural_sort_key(p.dataset), natural_sort_key(p.image.name)),
    )
    n = len(pairs)
    if n_train + n_val + n_test > n:
        raise ValueError(
            f"Zu wenige Bilder: {n} vorhanden, benötigt {n_train}+{n_val}+{n_test}="
            f"{n_train + n_val + n_test}. Mehr Sub-Datensätze wählen oder Größen reduzieren."
        )
    rng = random.Random(split_seed)
    if mode == "random":
        idx = list(range(n))
        rng.shuffle(idx)
        test = [pairs[i] for i in idx[:n_test]]
        val = [pairs[i] for i in idx[n_test : n_test + n_val]]
        pool = [pairs[i] for i in idx[n_test + n_val :]]
    elif mode == "contiguous":
        groups: dict[str, list[Pair]] = {}
        for p in pairs:
            groups.setdefault(p.dataset, []).append(p)
        test, val, pool = [], [], []
        rem_t, rem_v, rem_n = n_test, n_val, n
        for g in groups.values():
            m = len(g)
            t = round(rem_t * m / rem_n) if rem_n > 0 else 0
            v = round(rem_v * m / rem_n) if rem_n > 0 else 0
            test += g[m - t :]
            val += g[m - t - v : m - t]
            pool += g[: m - t - v]
            rem_t -= t
            rem_v -= v
            rem_n -= m
        rng.shuffle(pool)
    else:
        raise ValueError(f"Unbekannter split_mode: {mode}")
    if len(pool) < n_train:
        raise ValueError(f"Trainings-Pool ({len(pool)}) kleiner als n_train ({n_train}).")
    return {"train": pool[:n_train], "val": val, "test": test}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _read_tiff(path: Path) -> np.ndarray:
    a = np.squeeze(tifffile.imread(str(path)))
    if a.ndim == 3:  # RGB(A)/Multi-Page -> erster Kanal bzw. erste Seite
        a = a[..., 0] if a.shape[-1] in (2, 3, 4) else a[0]
    if a.ndim != 2:
        raise ValueError(f"{path}: unerwartete Form {a.shape}")
    return a


def _to_float01(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


DEFAULT_AUG = dict(flip=True, rot90=True, rot_deg=0.0, intensity=0.2)


class MitoDataset(Dataset):
    """Liefert dict(image [3,S,S] normalisiert, mask [1,S,S] in {0,1}, name)."""

    def __init__(self, pairs: list[Pair], img_size: int = 448, train: bool = False,
                 aug: dict | None = None, cache: bool = True):
        self.pairs, self.size, self.train = pairs, img_size, train
        self.aug = {**DEFAULT_AUG, **(aug or {})}
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        self._cache = [self._load(p) for p in pairs] if cache else None

    def _load(self, pair: Pair):
        img = torch.from_numpy(_to_float01(_read_tiff(pair.image)))[None]
        msk = torch.from_numpy((_read_tiff(pair.mask) > 0).astype(np.float32))[None]  # Instanzen -> binär
        size = [self.size, self.size]
        img = TF.resize(img, size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        msk = TF.resize(msk, size, interpolation=InterpolationMode.NEAREST)
        return img.clamp_(0, 1), msk

    def _augment(self, img: torch.Tensor, msk: torch.Tensor):
        a = self.aug
        if a["flip"]:
            if torch.rand(()) < 0.5:
                img, msk = img.flip(-1), msk.flip(-1)
            if torch.rand(()) < 0.5:
                img, msk = img.flip(-2), msk.flip(-2)
        if a["rot90"]:
            k = int(torch.randint(0, 4, ()))
            img, msk = torch.rot90(img, k, (-2, -1)), torch.rot90(msk, k, (-2, -1))
        if a["rot_deg"] > 0 and torch.rand(()) < 0.5:
            ang = float((torch.rand(()) * 2 - 1) * a["rot_deg"])
            img = TF.rotate(img, ang, interpolation=InterpolationMode.BILINEAR, fill=[float(img.mean())])
            msk = TF.rotate(msk, ang, interpolation=InterpolationMode.NEAREST, fill=[0.0])
        if a["intensity"] > 0 and torch.rand(()) < 0.5:  # leichter Kontrast-/Helligkeits-Jitter
            s = a["intensity"]
            c = 1 + (torch.rand(()) * 2 - 1) * s
            b = (torch.rand(()) * 2 - 1) * s * 0.5
            mu = img.mean()
            img = ((img - mu) * c + mu + b).clamp(0, 1)
        return img.contiguous(), msk.contiguous()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        img, msk = self._cache[i] if self._cache is not None else self._load(self.pairs[i])
        if self.train:
            img, msk = self._augment(img.clone(), msk.clone())
        img = (img.expand(3, -1, -1) - self.mean) / self.std
        return {"image": img, "mask": msk, "name": self.pairs[i].name}


# --------------------------------------------------------------------------- #
# DataLoader-Fabrik
# --------------------------------------------------------------------------- #
def _worker_init(worker_id: int):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def get_dataloaders(cfg: dict):
    """Baut train/val/test-Loader aus einer Config. Gibt (loaders, info) zurück.

    info enthält Bildanzahlen und die Dateilisten aller drei Splits (train/val/test).
    """
    pairs = discover_pairs(cfg["data_root"], cfg.get("datasets"))
    splits = make_splits(pairs, cfg["train_size"], cfg["val_size"], cfg["test_size"],
                         cfg.get("split_seed", 0), cfg.get("split_mode", "contiguous"))
    aug = dict(flip=cfg.get("aug_flip", True), rot90=cfg.get("aug_rot90", True),
               rot_deg=cfg.get("aug_rot_deg", 0.0), intensity=cfg.get("aug_intensity", 0.2))
    cache = cfg.get("cache_data", True)
    nw = cfg.get("num_workers", 4)
    gen = torch.Generator().manual_seed(cfg.get("seed", 0))

    def make(name, train):
        ds = MitoDataset(splits[name], cfg["img_size"], train=train, aug=aug, cache=cache)
        return DataLoader(ds, batch_size=cfg["batch_size"] if train else max(cfg["batch_size"], 8),
                          shuffle=train, num_workers=nw, pin_memory=torch.cuda.is_available(),
                          persistent_workers=nw > 0, worker_init_fn=_worker_init,
                          generator=gen if train else None)

    loaders = {k: make(k, k == "train") for k in ("train", "val", "test")}
    info = {
        "n_total_available": len(pairs),
        "n": {k: len(v) for k, v in splits.items()},
        "datasets_in_train": sorted({p.dataset for p in splits["train"]}),
        "train_files": [p.name for p in splits["train"]],
        "val_files": [p.name for p in splits["val"]],
        "test_files": [p.name for p in splits["test"]],
    }
    return loaders, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sub-Datensätze und Bildanzahlen auflisten")
    ap.add_argument("--root", required=True)
    ap.add_argument("--min_images", type=int, default=0)
    a = ap.parse_args()
    counts = {k: v for k, v in list_datasets(a.root).items() if v >= a.min_images}
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{v:6d}  {k}")
    print(f"-> {len(counts)} Sub-Datensätze, {sum(counts.values())} Bilder")
