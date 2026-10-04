# Image Captioning on Flickr8k — CNN Encoder + LSTM / Transformer Decoder, with Prompt-Guided Control

Final project for the Neural Networks & Deep Learning course. A full image
captioning pipeline is built on top of a single, shared preprocessing/feature
cache (Flickr8k, ResNet50 frozen features), with four trained models on top of it:

1. **Part 1 — Baseline**: frozen ResNet50 + single-layer LSTM decoder (no spatial attention).
2. **Part 2 — Transformer decoder**: same frozen ResNet50 features, LSTM replaced
   by a Transformer decoder with masked self-attention and cross-attention over
   the image's 49 spatial feature tokens.
3. **Part 3 — Prompt-guided captioning**: the LSTM and the Transformer decoder
   each extended with a 6-way tag embedding (`<GENERAL>`, `<OBJECTS>`, `<ACTION>`,
   `<ENVIRONMENT>`, `<SHORT>`, `<DETAILED>`) so caption style/content can be
   controlled at generation time, warm-started from the Part 1 / Part 2 checkpoints.

The CNN encoder is frozen and run **once**, in the preprocessing notebook; every
other notebook trains only a lightweight decoder on cached features. All five
notebooks below were executed end-to-end on Kaggle (T4/P100) with no errors.

## Results

All metrics are **test-set** BLEU, one generated caption per image compared
against all available human references for that image.

| Model | Decoder | Test BLEU-1 | Test BLEU-4 | Notes |
|---|---|---|---|---|
| Part 1 — Baseline | 1-layer LSTM, no attention | **0.598** | **0.1969** | greedy decoding, best epoch 17/23 |
| Part 2 — Transformer | Cross-attention decoder, greedy | 0.5963 | 0.1942 | best epoch 12/20, slightly below Part 1 |
| Part 2 — Transformer | Cross-attention decoder, beam-3 | **0.6099** | **0.1985** | secondary decoding analysis; edges past Part 1 |
| Part 3 — Prompt-guided LSTM | Tag-conditioned LSTM | 0.3661 | 0.0968 | lower BLEU is expected — see discussion below |
| Part 3 — Prompt-guided Transformer | Tag-conditioned Transformer | 0.3622 | 0.0948 | same tag set, warm-started from Part 2 |

Per-tag breakdown, training curves, and all other raw numbers behind this table
are in `results/` (see structure below) — they were extracted directly from the
executed notebooks' saved outputs, not retyped by hand.

### Discussion

- **Part 2 vs Part 1 (fair, greedy comparison):** the Transformer decoder ends
  up essentially tied with the LSTM baseline on greedy test BLEU-4 (0.1942 vs
  0.1969 — a 0.003 gap), despite validation BLEU-4 peaking noticeably higher
  for the Transformer (0.210 vs 0.199). This is consistent with published
  results on Flickr8k: the dataset is small (~6,500 training images), which is
  known to make Transformer decoders harder to push past well-tuned LSTM
  baselines without additional data or regularization tricks ([see tuning notes
  used for this project](https://github.com/mikkkeldp/transformer-image-captioner)).
  With **beam search (width 3)**, the Transformer's BLEU-4 (0.1985) does edge
  past the LSTM's greedy BLEU-4 — reported as a secondary result since Part 1
  uses greedy decoding for its official number.
- **Part 3 tag control actually works:** for the `<SHORT>` vs `<DETAILED>` tag
  pair, the generated caption is longer under `<DETAILED>` than under `<SHORT>`
  for the **same image** in 86.7% of test images (LSTM) and 94.3% of test images
  (Transformer) — direct evidence that the prompt/tag conditioning is
  influencing generation, not being ignored by the model.
- **Why Part 3's BLEU is lower than Part 1/2:** Part 3's captions are
  tag-specific (e.g. `<OBJECTS>` captions list only objects, `<SHORT>` captions
  are deliberately brief), so they diverge from the generic, unconstrained
  human reference captions used for BLEU scoring. The `<OBJECTS>` tag alone
  scores much higher (BLEU-4 ~0.41-0.43) than the overall average, showing
  BLEU here is tag-dependent rather than a sign of a worse model.

### Known issue (cosmetic, does not affect results)

In both Part 3 notebooks, the per-epoch progress `print(...)` uses an
f-string with doubled braces (`f"...{{epoch+1:02d}}..."`), so the printed
training log shows the literal template text instead of the numbers. This
**only affects the printed log** — `history_df` / `training_history.csv` and
the loss/BLEU curve plots use the underlying variables directly and are
correct. Worth a one-line fix (`{epoch+1:02d}` instead of `{{epoch+1:02d}}`)
before the next run.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_preprocessing.ipynb          # downloads data, builds vocab, caches ResNet50 features
│   ├── 02_part1_baseline_cnn_lstm.ipynb     # Part 1: CNN + LSTM baseline
│   ├── 03_part2_transformer_decoder.ipynb   # Part 2: Transformer decoder + cross-attention
│   ├── 04_part3_prompt_guided_lstm.ipynb    # Part 3: tag-guided LSTM
│   └── 05_part3_prompt_guided_transformer.ipynb  # Part 3: tag-guided Transformer
└── results/                                  # extracted from the executed notebooks above
    ├── part0_preprocessing/
    │   ├── manifest.json
    │   ├── preprocessing_audit.json
    │   ├── transform_config.json
    │   └── train_batch_grid_5x5.png
    ├── part1_baseline/
    │   ├── hyperparameters_table.csv
    │   ├── test_metrics.json
    │   ├── training_history.csv
    │   ├── loss_curve.png
    │   ├── validation_bleu_curve.png
    │   ├── test_sample_generations.png
    │   └── failure_cases_for_manual_analysis.png
    ├── part2_transformer/
    │   ├── hyperparameters_table.csv
    │   ├── test_metrics.json
    │   ├── training_history.csv
    │   ├── loss_curve.png
    │   ├── validation_bleu_curve.png
    │   ├── epoch_time_curve.png
    │   ├── transformer_test_examples.png
    │   ├── transformer_failure_cases.png
    │   ├── lstm_vs_transformer_validation_bleu4.png
    │   ├── lstm_vs_transformer_caption_examples.png
    │   └── attention_maps/attention_example_{1..4}.png
    └── part3_prompt_guided/
        ├── lstm/        (test_metrics.json, per_tag_test_metrics.csv, length_control_by_tag.csv,
        │                 short_vs_detailed_control_check.json, same_image_different_tags_examples.txt,
        │                 loss_curve.png, validation_bleu4_curve.png)
        └── transformer/ (same files as above)
```

(`results.zip` in this delivery is this whole folder, zipped — unzip it into `results/` at the repo root.)

## What is *not* in this repository (and why)

| Artifact | Why it's excluded | Where it lives instead |
|---|---|---|
| Raw Flickr8k images | Copyright + size | Downloaded by notebook 01 (Kaggle dataset `adityajn105/flickr8k`) |
| `features.h5` (~149 MB) | Regenerable, large | Kaggle notebook 01 output |
| Model checkpoints (`*.pt`, tens-to-hundreds of MB each) | GitHub's 100 MB/file limit | Kaggle notebook outputs |
| Presentation ZIP bundles | Large, duplicates checkpoints | Kaggle notebook outputs |

`results/` intentionally keeps only small, inspectable artifacts (metrics,
tables, plots) extracted from the notebooks' own outputs.

## How to run

Run on **Kaggle Notebooks** (free GPU, T4/P100), in this order, attaching each
previous notebook's published output as input to the next:

1. `01_data_preprocessing.ipynb` — attach the `adityajn105/flickr8k` Kaggle
   dataset via **Add Input** first; the notebook expects it at
   `/kaggle/input/datasets/adityajn105/flickr8k` (edit `CFG.INPUT_ROOT` if your
   dataset mounts at a different path).
2. `02_part1_baseline_cnn_lstm.ipynb` — attach notebook 1's output.
3. `03_part2_transformer_decoder.ipynb` — attach notebook 1's and notebook 2's output (needed for the LSTM-vs-Transformer comparison).
4. `04_part3_prompt_guided_lstm.ipynb` — attach notebook 1's and notebook 2's output (warm-starts from the Part 1 checkpoint).
5. `05_part3_prompt_guided_transformer.ipynb` — attach notebook 1's and notebook 3's output (warm-starts from the Part 2 checkpoint).

## Requirements

See `requirements.txt`. Core stack: PyTorch, torchvision, h5py, NLTK (BLEU),
pandas/numpy, matplotlib, tqdm, Pillow.


Academic coursework project. No specific license is applied; please contact the
author before reusing substantial parts of the code.
