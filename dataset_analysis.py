"""Vollständige Analyse eines CEM-MitoLab-artigen Datensatzes.

Beispiele:
    # Einen Datensatz analysieren:
    python dataset_analysis.py --data_root /pfad/cem_mitolab --dataset Bleck2018_SoleusBV2

    # Alle Sub-Datensätze analysieren:
    python dataset_analysis.py --data_root /pfad/cem_mitolab

    # Schnelltest mit den ersten 50 Bildern und 8 Prozessen:
    python dataset_analysis.py --data_root /pfad/cem_mitolab --dataset Bleck2018_SoleusBV2 \
        --max_images 50 --num_workers 8

Ausgaben pro Datensatz in <out_dir>/<dataset>/:
    per_image.csv        # eine Zeile pro Bild (Fläche, Instanzen, LOC, ROI, ...)
    per_loc.csv          # Zusammenfassung pro LOC
    per_roi.csv          # Zusammenfassung pro ROI
    area_histogram.csv   # Verteilung der Mitochondrienfläche (5-%-Bins)
    summary.json         # alle Kennzahlen maschinenlesbar
    summary.txt          # lesbare Zusammenfassung
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

TIFF_EXTS = (".tif", ".tiff")

PER_IMAGE_FIELDS = [
    "dataset", "image", "mask", "status", "error",
    "image_width", "image_height", "mask_width", "mask_height",
    "fg_pixels", "total_pixels", "area_percent",
    "n_instances", "mask_max", "mask_is_binary",
    "instance_area_px_min", "instance_area_px_median",
    "instance_area_px_mean", "instance_area_px_max",
    "roi", "loc", "slice",
]


def natural_sort_key(s: str) -> tuple[list[tuple[int, int | str]], str]:
    tokens = re.split(r"(\d+)", s)
    key: list[tuple[int, int | str]] = []
    for t in tokens:
        if not t:
            continue
        key.append((1, int(t)) if t.isdigit() else (0, t.lower()))
    return (key, s)


def load_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile
        a = np.squeeze(tifffile.imread(str(path)))
    except ImportError:
        from PIL import Image
        a = np.array(Image.open(str(path)))
    if a.ndim == 3:
        a = a[..., 0] if a.shape[-1] in (2, 3, 4) else a[0]
    if a.ndim != 2:
        raise ValueError(f"{path}: unerwartete Dimensionen {a.shape}")
    return a


def parse_meta(name: str) -> dict:
    """Zieht ROI, LOC und Slice aus dem Dateinamen."""
    roi = re.search(r"ROI-(.+?)-LOC-", name)
    loc = re.search(r"LOC-(\d+)", name)
    sl = re.search(r"LOC-\d+_(\d+)", name)
    return {
        "roi": roi.group(1) if roi else "",
        "loc": loc.group(1) if loc else "",
        "slice": int(sl.group(1)) if sl else None,
    }


def list_datasets(root: Path) -> list[str]:
    return sorted(
        [d.name for d in root.iterdir()
         if d.is_dir() and (d / "images").is_dir() and (d / "masks").is_dir()],
        key=natural_sort_key,
    )


def discover_tasks(root: Path, dataset: str, max_images: int = 0) -> list[tuple]:
    img_dir, mask_dir = root / dataset / "images", root / dataset / "masks"
    imgs = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in TIFF_EXTS],
        key=lambda p: natural_sort_key(p.name),
    )
    if max_images and max_images > 0:
        imgs = imgs[:max_images]
    tasks = []
    for img in imgs:
        mask = mask_dir / img.name
        tasks.append((dataset, img, mask if mask.exists() else None))
    return tasks


def analyze_one(task: tuple) -> dict:
    dataset, img_path, mask_path = task
    meta = parse_meta(img_path.name)
    row = {
        "dataset": dataset, "image": img_path.name,
        "mask": mask_path.name if mask_path else "",
        "status": "ok", "error": "",
        "image_width": None, "image_height": None,
        "mask_width": None, "mask_height": None,
        "fg_pixels": None, "total_pixels": None, "area_percent": None,
        "n_instances": None, "mask_max": None, "mask_is_binary": None,
        "instance_area_px_min": None, "instance_area_px_median": None,
        "instance_area_px_mean": None, "instance_area_px_max": None,
        "roi": meta["roi"], "loc": meta["loc"], "slice": meta["slice"],
        "_instance_areas": [],
    }
    if mask_path is None:
        row["status"] = "missing_mask"
        return row
    try:
        img = load_tiff(img_path)
        mask = load_tiff(mask_path)
    except Exception as e:  # beschädigte Datei o. Ä.
        row["status"] = "error"
        row["error"] = str(e)
        return row

    row["image_width"], row["image_height"] = int(img.shape[1]), int(img.shape[0])
    row["mask_width"], row["mask_height"] = int(mask.shape[1]), int(mask.shape[0])

    fg = int((mask > 0).sum())
    total = int(mask.size)
    row["fg_pixels"] = fg
    row["total_pixels"] = total
    row["area_percent"] = round(100.0 * fg / total, 4) if total else 0.0

    mask_max = int(mask.max()) if mask.size else 0
    row["mask_max"] = mask_max
    row["mask_is_binary"] = mask_max <= 1

    if mask_max > 1:  # Instanz-Labels 1..N
        labels = np.unique(mask)
        n_inst = int(len(labels) - (1 if labels[0] == 0 else 0))
        areas = np.bincount(mask.ravel().astype(np.int64))[1:]
        areas = areas[areas > 0]
    else:             # binäre Maske: eine Instanz, falls Vordergrund
        n_inst = int(fg > 0)
        areas = np.array([fg], dtype=np.int64) if fg > 0 else np.array([], dtype=np.int64)

    row["n_instances"] = n_inst
    if len(areas):
        row["instance_area_px_min"] = int(areas.min())
        row["instance_area_px_median"] = float(np.median(areas))
        row["instance_area_px_mean"] = float(areas.mean())
        row["instance_area_px_max"] = int(areas.max())
        row["_instance_areas"] = [int(a) for a in areas]
    return row


def _stats(values: list) -> dict:
    vals = sorted(float(v) for v in values if v is not None)
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "min": None,
                "p05": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None}

    def pct(p: float) -> float:
        if n == 1:
            return vals[0]
        k = (n - 1) * p
        lo, hi = int(k), min(int(k) + 1, n - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

    return {
        "n": n,
        "mean": round(statistics.fmean(vals), 4),
        "std": round(statistics.pstdev(vals), 4) if n > 1 else 0.0,
        "min": round(vals[0], 4),
        "p05": round(pct(0.05), 4),
        "p25": round(pct(0.25), 4),
        "median": round(pct(0.5), 4),
        "p75": round(pct(0.75), 4),
        "p95": round(pct(0.95), 4),
        "max": round(vals[-1], 4),
    }


def group_summary(rows: list[dict], key: str) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        areas = [r["area_percent"] for r in rs]
        out[g] = {
            "n_images": len(rs),
            "n_empty": sum(1 for a in areas if a == 0),
            "area_percent_mean": round(statistics.fmean(areas), 4) if areas else 0.0,
            "area_percent_median": round(statistics.median(areas), 4) if areas else 0.0,
            "n_instances_total": sum(r["n_instances"] for r in rs),
        }
    return out


def summarize(rows: list[dict], tiny_area: float, large_area: float) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    areas = [r["area_percent"] for r in ok]
    nonempty = [a for a in areas if a > 0]
    inst_counts = [r["n_instances"] for r in ok]
    inst_areas = [a for r in ok for a in r["_instance_areas"]]

    return {
        "n_images": len(rows),
        "n_ok": len(ok),
        "n_missing_mask": sum(1 for r in rows if r["status"] == "missing_mask"),
        "n_errors": sum(1 for r in rows if r["status"] == "error"),
        "n_empty_masks": sum(1 for a in areas if a == 0),
        "pct_empty_masks": round(100.0 * sum(1 for a in areas if a == 0) / len(ok), 4) if ok else 0.0,
        "n_tiny_masks": sum(1 for a in areas if 0 < a < tiny_area),
        "n_large_masks": sum(1 for a in areas if a > large_area),
        "tiny_area_threshold": tiny_area,
        "large_area_threshold": large_area,
        "area_percent": _stats(areas),
        "area_percent_nonempty": _stats(nonempty),
        "n_instances_total": sum(inst_counts),
        "n_zero_instance_images": sum(1 for n in inst_counts if n == 0),
        "n_instances": _stats(inst_counts),
        "instance_area_px": _stats(inst_areas),
        "mask_is_binary_count": sum(1 for r in ok if r["mask_is_binary"]),
        "image_size_counts": dict(Counter(f"{r['image_width']}x{r['image_height']}" for r in ok)),
        "mask_size_counts": dict(Counter(f"{r['mask_width']}x{r['mask_height']}" for r in ok)),
        "per_loc": group_summary(ok, "loc"),
        "per_roi": group_summary(ok, "roi"),
    }


def write_per_image(path: Path, rows: list[dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PER_IMAGE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_group_csv(path: Path, groups: dict, key_name: str):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([key_name, "n_images", "n_empty", "area_percent_mean",
                         "area_percent_median", "n_instances_total"])
        for g, s in groups.items():
            writer.writerow([g, s["n_images"], s["n_empty"], s["area_percent_mean"],
                             s["area_percent_median"], s["n_instances_total"]])


def write_histogram(path: Path, areas: list[float], bin_width: float = 5.0):
    n_bins = int(100 / bin_width)
    bins = [i * bin_width for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for a in areas:
        counts[min(int(a // bin_width), n_bins - 1)] += 1
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_start_percent", "bin_end_percent", "count"])
        for i, c in enumerate(counts):
            writer.writerow([bins[i], bins[i + 1], c])


def write_summary_txt(path: Path, s: dict):
    a = s["area_percent"]
    an = s["area_percent_nonempty"]
    inst = s["n_instances"]
    ia = s["instance_area_px"]

    def fmt(d: dict, key: str) -> str:
        v = d.get(key)
        return "–" if v is None else f"{v:.2f}"

    lines = [
        f"Datensatz: {s['dataset']}",
        f"Bilder insgesamt: {s['n_images']}",
        f"  mit Maske: {s['n_ok']}",
        f"  fehlende Masken: {s['n_missing_mask']}",
        f"  Fehler: {s['n_errors']}",
        f"  leere Masken (0% Fläche): {s['n_empty_masks']} ({s['pct_empty_masks']:.2f}%)",
        f"  winzige Masken (<{s['tiny_area_threshold']}%): {s['n_tiny_masks']}",
        f"  große Masken (>{s['large_area_threshold']}%): {s['n_large_masks']}",
        "",
        "Mitochondrienfläche pro Bild (%):",
        f"  mean ± std: {fmt(a, 'mean')} ± {fmt(a, 'std')}",
        f"  min / p05 / median / p95 / max: "
        f"{fmt(a, 'min')} / {fmt(a, 'p05')} / {fmt(a, 'median')} / {fmt(a, 'p95')} / {fmt(a, 'max')}",
        f"  nur nicht-leere Bilder: mean {fmt(an, 'mean')}, median {fmt(an, 'median')}",
        "",
        "Instanzen:",
        f"  gesamt: {s['n_instances_total']}",
        f"  pro Bild: mean {fmt(inst, 'mean')}, median {fmt(inst, 'median')}, max {fmt(inst, 'max')}",
        f"  Bilder ohne Instanz: {s['n_zero_instance_images']}",
        f"  Instanzfläche (px): min {fmt(ia, 'min')}, median {fmt(ia, 'median')}, "
        f"mean {fmt(ia, 'mean')}, max {fmt(ia, 'max')}",
        "",
        f"Bildgrößen (Breite x Höhe): {s['image_size_counts']}",
        f"Maskengrößen (Breite x Höhe): {s['mask_size_counts']}",
        f"Binäre Masken: {s['mask_is_binary_count']}",
        "",
        "LOC-Zusammenfassung:",
    ]
    for loc, g in s["per_loc"].items():
        lines.append(f"  LOC-{loc}: {g['n_images']} Bilder, leer {g['n_empty']}, "
                     f"mean {g['area_percent_mean']:.2f}%, median {g['area_percent_median']:.2f}%, "
                     f"Instanzen {g['n_instances_total']}")
    lines += ["", "ROI-Zusammenfassung:"]
    for roi, g in s["per_roi"].items():
        lines.append(f"  {roi}: {g['n_images']} Bilder, leer {g['n_empty']}, "
                     f"mean {g['area_percent_mean']:.2f}%, median {g['area_percent_median']:.2f}%, "
                     f"Instanzen {g['n_instances_total']}")
    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True, help="Pfad zu cem_mitolab")
    p.add_argument("--dataset", default=None, help="Ein Sub-Datensatz; ohne Angabe werden alle analysiert")
    p.add_argument("--out_dir", default="dataset_analysis", help="Ausgabeordner (Default: dataset_analysis)")
    p.add_argument("--num_workers", type=int, default=4, help="Parallele Prozesse (Default: 4)")
    p.add_argument("--max_images", type=int, default=0, help="0 = alle Bilder (für Schnelltests)")
    p.add_argument("--tiny_area", type=float, default=0.1, help="Schwelle für 'winzige' Masken in %% (Default: 0.1)")
    p.add_argument("--large_area", type=float, default=80.0, help="Schwelle für 'große' Masken in %% (Default: 80)")
    args = p.parse_args()

    root = Path(args.data_root)
    if not root.is_dir():
        raise SystemExit(f"data_root nicht gefunden: {root}")

    datasets = [args.dataset] if args.dataset else list_datasets(root)
    if not datasets:
        raise SystemExit(f"Keine Sub-Datensätze unter {root} gefunden.")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_summaries: dict[str, dict] = {}

    for ds in datasets:
        tasks = discover_tasks(root, ds, args.max_images)
        if not tasks:
            print(f"[analysis] {ds}: keine Bilder gefunden, überspringe.")
            continue
        print(f"[analysis] {ds}: {len(tasks)} Bilder ...")

        if args.num_workers > 1:
            with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
                rows = list(ex.map(analyze_one, tasks, chunksize=8))
        else:
            rows = [analyze_one(t) for t in tasks]

        summary = summarize(rows, args.tiny_area, args.large_area)
        summary["dataset"] = ds

        ds_out = out_root / ds
        ds_out.mkdir(parents=True, exist_ok=True)
        write_per_image(ds_out / "per_image.csv", rows)
        write_group_csv(ds_out / "per_loc.csv", summary["per_loc"], "loc")
        write_group_csv(ds_out / "per_roi.csv", summary["per_roi"], "roi")
        write_histogram(ds_out / "area_histogram.csv", [r["area_percent"] for r in rows if r["status"] == "ok"])
        with open(ds_out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        write_summary_txt(ds_out / "summary.txt", summary)

        print(f"[analysis] {ds}: {summary['n_ok']} Bilder, "
              f"Mittel {summary['area_percent']['mean']:.2f}% Fläche, "
              f"0%-Bilder {summary['n_empty_masks']} -> {ds_out}/")
        all_rows += rows
        all_summaries[ds] = summary

    if len(datasets) > 1 and all_rows:
        write_per_image(out_root / "all_per_image.csv", all_rows)
        with open(out_root / "all_summary.json", "w") as f:
            json.dump(all_summaries, f, indent=2, ensure_ascii=False)
        print(f"[analysis] Gesamt -> {out_root}/all_summary.json")


if __name__ == "__main__":
    main()
