"""Konvertiert TIFF-Bilder und Masken in PNG zur einfachen Vorschau.

Erstellt praktische Side-by-Side-Bilder (Original | Maske | Overlay),
damit man sie sofort mit jedem Bildbetrachter oder Browser ansehen kann.
Zusätzlich wird preview_stats.csv mit der Mitochondrienfläche pro Bild geschrieben.

Mit --save_individual speichert es zusätzlich zu dem Kombibild auch _img.png, _mask.png und _overlay.png separat:
Standardmäßig die ersten N Bilder aus dem Datensatz, oder mit --random zufällige Bilder (ohne Seed) oder mit --names bestimmte Bilder.

Beispiele:
    # 5 Bilder aus einem Datensatz als Side-by-Side PNGs exportieren:
    python preview.py --data_root /home/ul/ul_student/ul_lxm75/workspaces/pfs7wor9/sdg_workspace/EMPIAR-11037/cem_mitolab --dataset Bleck2018_SoleusBV2 --num 5

    # Bestimmte Bilder nach Name exportieren:
    python preview.py --data_root /pfad/cem_mitolab --dataset Bleck2018_SoleusBV2 --names heart_image0.tiff heart_image1.tiff

    # Direkt auf einen Unterordner zeigen:
    python preview.py --data_root /pfad/cem_mitolab/Bleck2018_SoleusBV2 --num 3
    
    # Zusätzlich zu Kombibild auch Einzeldateien speichern:
    python preview.py --data_root ... --dataset ... --num 3 --save_individual

    # Nur die Statistik (Mitochondrien-Fläche) für bereits vorhandene Previews neu schreiben:
    python preview.py --report_only --out_dir preview_output
"""
from __future__ import annotations

import argparse
import csv
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


def mask_from_preview(path: Path) -> np.ndarray:
    """Liest die binäre Maske aus dem mittleren Panel eines _preview.png."""
    if Image is None:
        raise RuntimeError("Pillow wird benötigt, um vorhandene Preview-PNGs zu lesen.")
    arr = np.array(Image.open(path).convert("RGB"))
    h, width, _ = arr.shape
    w = (width - 6) // 3  # [Bild | 3px | Maske | 3px | Overlay]
    if w <= 0 or 3 * w + 6 != width:
        raise ValueError(f"{path}: unerwartetes Preview-Layout ({width}x{h})")
    return (arr[:, w + 3 : 2 * w + 3, 0] > 0).astype(np.uint8)


def area_stats(mask_bin: np.ndarray) -> tuple[int, int, float]:
    fg = int(mask_bin.sum())
    total = int(mask_bin.size)
    return fg, total, (100.0 * fg / total if total else 0.0)


STATS_FIELDS = ["image", "mask", "width", "height", "fg_pixels", "total_pixels",
                "mito_area_percent", "n_mitochondria"]


def write_stats_csv(rows: list[dict], path: Path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def find_mask(data_root: Path | None, dataset: str, base: str) -> Path | None:
    if data_root is None:
        return None
    for ext in TIFF_EXTS:
        p = data_root / dataset / "masks" / f"{base}{ext}"
        if p.exists():
            return p
    return None


def count_instances(mask: np.ndarray) -> int:
    mmax = int(mask.max()) if mask.size else 0
    if mmax > 1:
        vals = np.unique(mask)
        return int(len(vals) - (1 if vals[0] == 0 else 0))
    return int((mask > 0).sum() > 0)


def report_from_previews(out_dir: Path, dataset: str | None = None, data_root: Path | None = None):
    """Schreibt preview_stats.csv für bereits vorhandene _preview.png-Dateien.

    Mit --data_root werden zusätzlich die Instanzanzahlen aus den Original-Masken gelesen.
    """
    if dataset:
        dirs = [out_dir / dataset]
    elif any(out_dir.glob("*_preview.png")):
        dirs = [out_dir]
    else:
        dirs = sorted([d for d in out_dir.iterdir() if d.is_dir()], key=lambda p: natural_sort_key(p.name))

    all_rows = []
    for d in dirs:
        pngs = sorted(d.glob("*_preview.png"), key=lambda p: natural_sort_key(p.name))
        if not pngs:
            continue
        rows = []
        for png in pngs:
            base = png.name[: -len("_preview.png")]
            mask_bin = mask_from_preview(png)
            fg, total, pct = area_stats(mask_bin)
            mask_path = find_mask(data_root, d.name, base)
            n_mito = ""
            if mask_path is not None:
                try:
                    n_mito = count_instances(load_tiff(mask_path))
                except Exception as e:
                    print(f"  [warn] {mask_path}: {e}")
            rows.append({
                "image": base,
                "mask": mask_path.name if mask_path else "",
                "width": int(mask_bin.shape[1]),
                "height": int(mask_bin.shape[0]),
                "fg_pixels": fg,
                "total_pixels": total,
                "mito_area_percent": round(pct, 2),
                "n_mitochondria": n_mito,
            })
        write_stats_csv(rows, d / "preview_stats.csv")
        all_rows += [{"dataset": d.name, **r} for r in rows]
        mean = sum(r["mito_area_percent"] for r in rows) / len(rows)
        empty = sum(1 for r in rows if r["mito_area_percent"] == 0)
        print(f"[report] {d.name}: {len(rows)} Bilder, Mittel {mean:.2f}%, 0%-Bilder: {empty} "
              f"-> {d / 'preview_stats.csv'}")

    if all_rows:
        combined = out_dir / "preview_stats_all.csv"
        with open(combined, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset"] + STATS_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[report] Gesamt -> {combined}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default=None, help="Pfad zu cem_mitolab oder direkt zum Unterordner")
    p.add_argument("--dataset", default=None, help="Name des Sub-Datensatzes (z. B. Bleck2018_SoleusBV2)")
    p.add_argument("--names", nargs="*", default=None, help="Spezifische Bildnamen (z. B. heart_image0.tiff)")
    p.add_argument("--num", type=int, default=5, help="Anzahl zu konvertierender Bilder (Default: 5)")
    p.add_argument("--random", action="store_true", help="Zufällige Bilder auswählen statt erste N")
    p.add_argument("--out_dir", default="preview_output", help="Ausgabeordner (Default: preview_output)")
    p.add_argument("--save_individual", action="store_true", help="Zusätzlich Einzeldateien (_img, _mask, _overlay) speichern")
    p.add_argument("--report_only", action="store_true",
                   help="Keine neuen Previews erzeugen; nur preview_stats.csv für vorhandene _preview.png schreiben "
                        "(mit --data_root werden n_mitochondria aus den Originalmasken ergänzt)")
    args = p.parse_args()

    if Image is None:
        raise SystemExit(
            "Fehler: 'Pillow' ist in dieser Python-Umgebung nicht installiert.\n"
            "Bitte führe den Befehl im Container aus (z. B. 'python preview.py ...') oder installiere Pillow mit 'pip install pillow'."
        )

    if args.report_only:
        report_from_previews(Path(args.out_dir), args.dataset,
                             Path(args.data_root) if args.data_root else None)
        return

    if not args.data_root:
        p.error("--data_root ist erforderlich (außer bei --report_only).")

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

    out_dir = Path(args.out_dir) / ds_dir.name
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
    stats_rows = []

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

        mito_count = count_instances(raw_mask)
        fg, total, fg_percent = area_stats(mask_bin)
        stats_rows.append({
            "image": img_path.name,
            "mask": mask_path.name,
            "width": int(mask_bin.shape[1]),
            "height": int(mask_bin.shape[0]),
            "fg_pixels": fg,
            "total_pixels": total,
            "mito_area_percent": round(fg_percent, 2),
            "n_mitochondria": mito_count,
        })
        print(f"  ✓ {sbs_path.name:35s} (Mitochondrien: {fg_percent:.2f}% Fläche)")

    stats_path = out_dir / "preview_stats.csv"
    write_stats_csv(stats_rows, stats_path)
    print(f"\n[preview] Statistik gespeichert: {stats_path.resolve()}")
    print(f"[preview] Fertig! Vorschauen gespeichert in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
