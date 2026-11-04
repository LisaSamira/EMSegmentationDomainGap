"""W&B-Sweep pro Trainingsgröße (13, 25, 50, 100, 200, 400).

Suchraum: sweep_space.yaml (modular editierbar).

Alles in einem Prozess (sequentiell):
    python sweep.py --data_root /pfad/cem_mitolab --datasets "SubA,SubB" --count 25

Nur Sweeps anlegen und Agents separat starten (z. B. SLURM-Array, parallel):
    python sweep.py --create_only ...      # gibt `wandb agent <entity/project/id>` je Größe aus

Feste Hyperparameter überschreiben:  --set epochs=50 --set img_size=224 --set split_mode=contiguous
"""
from __future__ import annotations

import argparse

import wandb
import yaml

import train


def load_space(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_sweep_config(space: dict, train_size: int, fixed: dict) -> dict:
    params = dict(space["parameters"])
    params.update(space.get("per_train_size", {}).get(train_size, {}))  # größenspezifische Überschreibungen
    params.update({k: {"value": v} for k, v in fixed.items() if v is not None})
    params["train_size"] = {"value": train_size}
    cfg = {
        "name": f"datasize-{train_size}",
        "method": space.get("method", "bayes"),
        "metric": space["metric"],
        "parameters": params,
        "program": "train.py",  # für `wandb agent` (CLI-Modus)
    }
    if "early_terminate" in space:
        cfg["early_terminate"] = space["early_terminate"]
    return cfg


def parse_set(items: list[str]) -> dict:
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        if k not in train.DEFAULTS:
            raise KeyError(f"--set {k}: unbekannter Key. Erlaubt: {list(train.DEFAULTS)}")
        d = train.DEFAULTS[k]
        out[k] = train._str2bool(v) if isinstance(d, bool) else (type(d)(v) if d is not None else v)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", default="sweep_space.yaml")
    ap.add_argument("--train_sizes", type=int, nargs="+", default=[13, 25, 50, 100, 200, 400])
    ap.add_argument("--count", type=int, default=25, help="Läufe pro Sweep (pro Trainingsgröße)")
    ap.add_argument("--data_root", default=train.DEFAULTS["data_root"])
    ap.add_argument("--datasets", default=train.DEFAULTS["datasets"], help='"all" oder "A,B,C"')
    ap.add_argument("--project", default=train.DEFAULTS["wandb_project"])
    ap.add_argument("--entity", default=None)
    ap.add_argument("--create_only", action="store_true")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", help="feste Config-Werte")
    a = ap.parse_args()

    space = load_space(a.space)
    fixed = {"data_root": a.data_root, "datasets": a.datasets,
             "wandb_project": a.project, "wandb_entity": a.entity, **parse_set(a.set)}

    sweep_ids = {}
    for n in a.train_sizes:
        sid = wandb.sweep(build_sweep_config(space, n, fixed), project=a.project, entity=a.entity)
        sweep_ids[n] = sid
        print(f"[sweep] train_size={n}: {sid}")
        if a.create_only:
            ent = a.entity or "<entity>"
            print(f"        wandb agent {ent}/{a.project}/{sid} --count {a.count}")
            continue
        wandb.agent(sid, function=train.run, count=a.count, project=a.project, entity=a.entity)

    print("\nSweep-IDs:", sweep_ids)


if __name__ == "__main__":
    main()
