"""Konvertiert TIFF-Bilder und Masken in PNG zur einfachen Vorschau.

Erstellt praktische Side-by-Side-Bilder (Original | Maske | Overlay),
damit man sie sofort mit jedem Bildbetrachter oder Browser ansehen kann.

Mit --save_individual speichert es zusätzlich zu dem Kombibild auch _img.png, _mask.png und _overlay.png separat:


Beispiele:
    # 5 Bilder aus einem Datensatz als Side-by-Side PNGs exportieren:
    python preview.py --data_root /pfad/cem_mitolab --dataset Bleck2018_SoleusBV2 --num 5

    # Bestimmte Bilder nach Name exportieren:
    python preview.py --data_root /pfad/cem_mitolab --dataset Bleck2018_SoleusBV2 --names heart_image0.tiff heart_image1.tiff

    # Direkt auf einen Unterordner zeigen:
    python preview.py --data_root /pfad/cem_mitolab/Bleck2018_SoleusBV2 --num 3
    
    # Zusätzlich zu Kombibild auch Einzeldateien speichern:
    python preview.py --data_root ... --dataset ... --num 3 --save_individual
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
try:
    from PIL import Image
except ImportError:
    Image = None

TIFF_EXTS = (".tif", ".tiff")


def natural_sort_key(s: str) -> tuple[list[tuple[int, int | str]], str]:
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


def load_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile
        a = np.squeeze(tifffile.imread(str(path)))
    except ImportError:
        a = np.array(Image.open(str(path)))
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] in (2, 3, 4) else a[0]
    if a.ndim != 2:
        raise ValueError(f"{path}: unerwartete Dimensionen {a.shape}")
    return a


def norm_image(img: np.ndarray) -> np.ndarray:
    """Normalisiert ein Graustufenbild robust auf 0..255 uint8."""
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi > lo:
        img = (img - lo) / (hi - lo) * 255.0
    else:
        img = np.zeros_like(img)
    return img.clip(0, 255).astype(np.uint8)


def create_overlay(img_u8: np.ndarray, mask_bin: np.ndarray, color=(255, 40, 40), alpha=0.45) -> np.ndarray:
    """Erzeugt ein RGB-Overlay: Maske farbig transparent über dem Graustufenbild."""
    rgb = np.stack([img_u8, img_u8, img_u8], axis=-1).astype(np.float32)
    fg = mask_bin > 0
    for c in range(3):
        rgb[fg, c] = rgb[fg, c] * (1.0 - alpha) + color[c] * alpha
    return rgb.clip(0, 255).astype(np.uint8)


def create_side_by_side(img_u8: np.ndarray, mask_bin: np.ndarray, overlay_u8: np.ndarray) -> np.ndarray:
    """Fügt Original | Maske | Overlay horizontal zusammen mit Trennstrichen."""
    img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)
    mask_u8 = (mask_bin * 255).astype(np.uint8)
    mask_rgb = np.stack([mask_u8, mask_u8, mask_u8], axis=-1)

    h, _, _ = img_rgb.shape
    sep = np.full((h, 3, 3), 200, dtype=np.uint8)  # heller Trennstreifen
    return np.concatenate([img_rgb, sep, mask_rgb, sep, overlay_u8], axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True, help="Pfad zu cem_mitolab oder direkt zum Unterordner")
    p.add_argument("--dataset", default=None, help="Name des Sub-Datensatzes (z. B. Bleck2018_SoleusBV2)")
    p.add_argument("--names", nargs="*", default=None, help="Spezifische Bildnamen (z. B. heart_image0.tiff)")
    p.add_argument("--num", type=int, default=5, help="Anzahl zu konvertierender Bilder (Default: 5)")
    p.add_argument("--random", action="store_true", help="Zufällige Bilder auswählen statt erste N")
    p.add_argument("--out_dir", default="preview_output", help="Ausgabeordner (Default: preview_output)")
    p.add_argument("--save_individual", action="store_true", help="Zusätzlich Einzeldateien (_img, _mask, _overlay) speichern")
    args = p.parse_args()

    if Image is None:
        raise SystemExit(
            "Fehler: 'Pillow' ist in dieser Python-Umgebung nicht installiert.\n"
            "Bitte führe den Befehl im Container aus (z. B. 'python preview.py ...') oder installiere Pillow mit 'pip install pillow'."
        )

    root = Path(args.data_root)
    if args.dataset:
        ds_dir = root / args.dataset
    elif (root / "images").is_dir():
        ds_dir = root
    else:
        subdirs = [d.name for d in root.iterdir() if d.is_dir() and (d / "images").is_dir()]
        raise ValueError(
            f"Bitte --dataset angeben. Gefundene Sub-Datensätze unter {root}:\n" + "\n".join(f"  - {s}" for s in subdirs[:15])
        )

    img_dir, mask_dir = ds_dir / "images", ds_dir / "masks"
    if not img_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Ordner 'images' oder 'masks' fehlt unter {ds_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bildpaare ermitteln
    if args.names:
        img_files = [img_dir / n for n in args.names]
    else:
        all_imgs = sorted(
            [p for p in img_dir.iterdir() if p.suffix.lower() in TIFF_EXTS and (mask_dir / p.name).exists()],
            key=lambda p: natural_sort_key(p.name)
        )
        if not all_imgs:
            raise RuntimeError(f"Keine passenden Bild/Masken-Paare in {ds_dir} gefunden.")
        if args.random:
            img_files = random.sample(all_imgs, min(args.num, len(all_imgs)))
        else:
            img_files = all_imgs[:args.num]

    print(f"[preview] Konvertiere {len(img_files)} Bilder aus {ds_dir.name} -> {out_dir}/ ...")

    for img_path in img_files:
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            print(f"  [warn] Keine Maske gefunden für {img_path.name}, überspringe.")
            continue

        raw_img = load_tiff(img_path)
        raw_mask = load_tiff(mask_path)

        img_u8 = norm_image(raw_img)
        mask_bin = (raw_mask > 0).astype(np.uint8)
        overlay_u8 = create_overlay(img_u8, mask_bin)

        base_name = img_path.stem
        # Side-by-Side: [Original | Maske | Overlay]
        sbs = create_side_by_side(img_u8, mask_bin, overlay_u8)
        sbs_path = out_dir / f"{base_name}_preview.png"
        Image.fromarray(sbs).save(sbs_path)

        if args.save_individual:
            Image.fromarray(img_u8).save(out_dir / f"{base_name}_img.png")
            Image.fromarray((mask_bin * 255).astype(np.uint8)).save(out_dir / f"{base_name}_mask.png")
            Image.fromarray(overlay_u8).save(out_dir / f"{base_name}_overlay.png")

        mito_count = int(raw_mask.max()) if raw_mask.max() > 1 else int(mask_bin.sum() > 0)
        fg_percent = (mask_bin.sum() / mask_bin.size) * 100
        print(f"  ✓ {sbs_path.name:35s} (Mitochondrien: {fg_percent:.1f}% Fläche)")

    print(f"\n[preview] Fertig! Vorschauen gespeichert in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
