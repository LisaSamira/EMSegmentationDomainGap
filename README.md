# Datensatzgrößen-Test: Mitochondrien-Segmentierung (2D-EM) mit frozen DINOv3

Initialer Trainingscode für die Masterarbeit *„What makes EM datasets different?“*. Ziel dieses Codes: testen, wie gut verschiedene Trainingsgrößen (**15, 25, 50, 100, 200, 400 Bilder**) funktionieren, inklusive Hyperparameter-Sweep pro Größe mit Weights & Biases.

- **Task:** binäre Segmentierung von Mitochondrien in 2D
- **Modell:** eingefrorenes DINOv3-Backbone + trainierbarer leichter Conv-Head
- **Metriken:** Foreground-IoU und Dice, dazu detaillierte Zeitmessung (Training, Validierung, Test)

## Dateien

| Datei | Inhalt |
| --- | --- |
| `dataset.py` | TIFF-Loader, Train/Val/Test-Splits, Augmentationen, DataLoader |
| `model.py` | Backbone + Light Head, `MODEL_REGISTRY` für weitere Varianten |
| `train.py` | Epochen-Training, Evaluation, Zeitmessung, W&B-Logging, CLI |
| `sweep.py` | legt pro Trainingsgröße einen W&B-Sweep an und startet Agents |
| `sweep_space.yaml` | Suchraum des Sweeps (hier anpassen) |
| `preview.py` | Konvertiert TIFFs + Masken nach PNG (Side-by-Side Vorschau) |
| `requirements.txt` | Abhängigkeiten |

## Installation

```bash
pip install -r requirements.txt
wandb login
huggingface-cli login   # DINOv3 ist ein gated Modell: Zugriff vorher auf HuggingFace beantragen
```

Benötigt wird `transformers>=4.56` (DINOv3-Unterstützung).

## Datenstruktur

Erwartet wird das bereits in 2D-TIFFs konvertierte CEM-MitoLab (EMPIAR-11037). `--data_root` zeigt auf den Ordner `cem_mitolab`:

```
cem_mitolab/
├── Bleck2018_SoleusBV2/
│   ├── images/<name>.tiff
│   └── masks/<name>.tiff      # gleicher Dateiname wie das Bild
├── Wei2020_MitoEM-H/
└── ...
```

- Bilder ohne gleichnamige Maske werden übersprungen (mit Warnung).
- Masken werden binarisiert (`> 0`), die Instanz-Labels also zu einer Vordergrundmaske zusammengefasst.
- Bilder werden auf `img_size × img_size` skaliert (Default 448, Vielfaches von 16). Nicht-quadratische Bilder werden dabei leicht verzerrt.

### Sub-Datensätze auswählen

```bash
python dataset.py --root /pfad/cem_mitolab --min_images 450
```

Listet alle Sub-Datensätze mit mindestens 450 Bildern auf (in CEM-MitoLab gibt es genau 10 passende Datensätze). Für 400 Trainingsbilder werden mindestens `400 + val_size + test_size` Bilder benötigt (Default 450).

Um Domänen-Eigenschaften und Unterschiede sauber zu erforschen (*„What makes EM datasets different?“*), werden Datensätze **niemals vermischt/gepoolt**, sondern jeder Datensatz wird separat trainiert und evaluiert: `--datasets "<SubDataset>"`.

## Schnellstart

```bash
# Einzellauf (Test, ohne Online-Logging)
python train.py --data_root /pfad/cem_mitolab --datasets "Bleck2018_SoleusBV2" \
                --train_size 25 --wandb_mode offline

# Sweeps für alle Größen für einen bestimmten Datensatz
python sweep.py --data_root /pfad/cem_mitolab --datasets "Bleck2018_SoleusBV2" --count 25

# Nur bestimmte Größen
python sweep.py --data_root /pfad/cem_mitolab --datasets "Bleck2018_SoleusBV2" --train_sizes 15 25 --count 10
```

Alle Schlüssel aus `DEFAULTS` in `train.py` sind als `--key wert` überschreibbar, z. B. `--lr 3e-4 --batch_size 16 --img_size 224`. Boolesche Werte: `--amp false`.

### Sweeps parallel auf dem Cluster

```bash
python sweep.py --create_only --data_root ... --datasets "..." --project mito-em-datasize --entity <entity>
# gibt pro Trainingsgröße aus:  wandb agent <entity>/<project>/<sweep_id> --count 25
```

Die `wandb agent`-Befehle können als separate SLURM-Jobs laufen. Auch mehrere Agents auf denselben Sweep sind möglich.

Feste Werte für alle Sweeps setzt du mit `--set`:

```bash
python sweep.py ... --set img_size=224 --set patience=0
```

## Datensplit

- **Val und Test sind fest** (`val_size`, `test_size`, Default je 25) und hängen nicht von `train_size` ab.
- **Trainingsmengen sind geschachtelt:** `train(15) ⊂ train(25) ⊂ … ⊂ train(400)`, solange `split_seed` gleich bleibt. Unterschiede zwischen Größen sind so echte Größeneffekte.
- `split_mode`:
  - `contiguous` (Standard): Val/Test werden pro Sub-Datensatz als zusammenhängende Blöcke vom Ende der Schnittserie genommen. Verhindert Z-Slice Data-Leakage bei 3D-Volumendaten (FIB-SEM/SBF-SEM), da benachbarte Schnitte nicht zwischen Train und Test aufgeteilt werden.
  - `random`: globales Mischen (führt bei 3D-Volumendaten zu künstlich überhöhten Scores durch Nachbarschnitt-Leakage).

## Modell

`DinoV3Segmenter` (`model.py`):

- Backbone: `facebook/dinov3-vitl16-pretrain-lvd1689m` (über `--backbone_name` änderbar), vollständig eingefroren und immer im `eval`-Modus. CLS- und Register-Tokens werden entfernt, die Patch-Tokens zu einer Feature-Map umgeformt.
- `n_feature_layers`: `1` = letzte Schicht, `4` = vier gleichmäßig verteilte Schichten, konkateniert.
- Head (`LightConvHead`): BatchNorm → 1×1-Conv → 3×3-Conv → Dropout → 2× (bilineares Upsampling ×2 + 3×3-Conv) → 1×1-Conv → bilineares Upsampling auf Eingabegröße. Gibt Logits aus.
- Nur der Head wird trainiert und gespeichert (`best_head.pt`).

### Weitere Modelle ergänzen

Neue Klasse mit `forward(x) -> Logits [B,1,H,W]`, `trainable_parameters()` und `trainable_state_dict()` schreiben, in `MODEL_REGISTRY` eintragen und mit `--model <name>` auswählen. In `train.py` wird der Head beim Laden des besten Checkpoints über `model.head` angesprochen. Hat eine neue Variante keinen `head`-Attribut, muss diese Stelle angepasst werden.

## Training und Evaluation

- AdamW, Cosine-Schedule mit linearem Warmup, Gradient Clipping, bf16-Autocast (nur auf GPU)
- Loss: BCE + `dice_weight` × Dice-Loss
- Bestes Modell nach **Val-IoU**; Early Stopping über `patience` (Default 20, `0` = aus)
- Nach dem Training wird der beste Checkpoint einmal auf dem **Testset** ausgewertet

### Metriken

- `iou`, `dice`: pixelweise über das gesamte Set (Mikro-Mittel), das ist die Hauptmetrik
- `iou_img`, `dice_img`: Mittel über Einzelbilder (leere GT und leere Vorhersage zählt als 1.0)

Mit Schwelle `--threshold 0.5`. Gemessen wird auf der skalierten Auflösung (`img_size`), nicht auf der Originalgröße.

### Zeitmessung

Alle Zeiten sind CUDA-synchronisiert (`torch.cuda.synchronize`) und werden pro Epoche nach W&B geloggt, die Summen zusätzlich in die Run-Summary:

| Key | Bedeutung |
| --- | --- |
| `time/setup_data_s`, `time/setup_model_s` | Laden der Daten bzw. des Modells |
| `time/train_epoch_s` | Dauer einer Trainingsepoche |
| `time/train_dataload_wait_s` | Zeit, die auf den DataLoader gewartet wurde |
| `time/val_s` | Validierungsdauer pro Epoche |
| `time/train_total_s`, `time/train_mean_epoch_s` | Gesamt- bzw. mittlere Trainingszeit (ohne Validierung) |
| `time/test_s`, `time/test_ms_per_image` | Testdauer gesamt und pro Bild |
| `time/run_total_s` | gesamte Laufzeit |
| `gpu_peak_mem_gb` | maximaler GPU-Speicher |

Bei aktivem Early Stopping hängt die Trainingszeit von der Zahl der gelaufenen Epochen ab (`epochs_run`). Für exakt vergleichbare Zeiten `--patience 0` setzen.

## Sweep-Konfiguration

Der Suchraum steht in `sweep_space.yaml` und wird ohne Codeänderung angepasst:

- `method`: `grid`, `random` oder `bayes` (Default)
- `metric`: `val/best_iou` (monoton steigendes Maximum der Val-IoU), niemals Test
- `early_terminate`: Hyperband
- `parameters`: Lernrate, Weight Decay, Batchgröße, Head-Kanäle, Dropout, Dice-Gewicht, Anzahl Feature-Layer
- `per_train_size`: Überschreibungen pro Trainingsgröße. Standardmäßig die Epochenzahl (200/150/100/60/40/30), damit die Zahl der Optimierungsschritte für kleine Sets vergleichbar bleibt.

Für jede Trainingsgröße wird ein eigener Sweep angelegt (`datasize-<n>`), Runs werden in W&B nach `train_size_<n>` gruppiert.

## Ausgabe

```
runs/n<train_size>_<wandb_run_id>/
├── best_head.pt     # Gewichte des Heads
└── results.json     # Config, Split-Info, beste Val-Werte, Test-Metriken, Zeiten
```

`results.json` enthält auch die Dateinamen der verwendeten Trainingsbilder.

## Bekannte Einschränkungen

- Pro Sweep-Lauf ein Seed. Für die finale Bewertung der besten Konfiguration (3 Seeds laut Konzept) ist noch keine Automatisierung enthalten.
- Bei 15 Trainingsbildern und einer Val-Menge von 25 Bildern ist die Val-IoU verrauscht, die Sweep-Ergebnisse für kleine Größen sind entsprechend mit Vorsicht zu lesen. `val_size` lässt sich erhöhen.
- Die Pipeline ist bisher nur für Training und Test auf derselben Datenmenge ausgelegt. `evaluate()` in `train.py` ist aber so gebaut, dass es später für Cross-Dataset-Tests (N×N-Matrix) wiederverwendet werden kann.
- Der Code wurde bislang nur syntaktisch geprüft (`py_compile`), nicht mit echten Daten und GPU ausgeführt.
