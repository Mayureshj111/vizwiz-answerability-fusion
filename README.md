# Flagging Unanswerable Visual Questions with Image–Text Fusion

Blind users of VizWiz photograph objects and ask spoken questions, but many questions
can't be answered (blurry, mis-framed or missing object). This project predicts whether
a question is **unanswerable**, so the app can prompt a retake instead of giving a wrong
answer. It compares unimodal baselines with early, late, cross-attention and gated fusion,
using CLIP as a reference.

## Dataset
VizWiz-VQA (CC BY 4.0): https://vizwiz.org/tasks-and-datasets/vqa/
Images and data are **not** included in this repository. The notebook downloads them.

| Split used | Source | Pairs | % unanswerable |
|---|---|---|---|
| Train | 85% of official train | 17,444 | 27.0 |
| Dev (tuning) | 15% of official train | 3,079 | 27.0 |
| Test | official validation | 4,319 | 32.1 |

The official test labels are hidden, so the official validation set is used as the test set.

## How to reproduce
1. Open `VizWiz_Fusion.ipynb` in Google Colab (button at the top of the notebook).
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all** and allow Google Drive access.
   Features and results are cached to `MyDrive/vizwiz/`.

Runtime on a free T4: about 15 min to download, ~12 min feature extraction,
<1 min per model (3 seeds each), ~12 min for CLIP features.

**Requirements:** Colab default environment (PyTorch, transformers, scikit-learn,
OpenCV, pandas, matplotlib). Random seeds: 0, 1, 2.

## Pipeline
- Images: EXIF rotation fix, 224×224 resize, frozen ViT-B/16 → 50 region tokens
- Questions: frozen BERT-base → 32 word tokens
- Loss: class-weighted BCE with soft labels; early stopping on dev AUROC;
  threshold tuned on dev

## Results (test, mean of 3 seeds)
| Model | Macro-F1 | AUROC | AUPRC |
|---|---|---|---|
| Question-only | 0.650 | 0.729 | 0.552 |
| Image-only | 0.696 | 0.772 | 0.617 |
| Late fusion | 0.730 | 0.827 | 0.681 |
| Early fusion | 0.762 | 0.864 | 0.745 |
| Cross-attention | 0.763 | 0.863 | 0.745 |
| Gated + modality dropout | 0.759 | 0.858 | 0.736 |
| CLIP reference | 0.801 | 0.900 | 0.805 |

All fusion models significantly beat image-only (95% bootstrap CIs exclude zero;
cross-attention +0.067 [0.056, 0.079]). Median latency: 95.9 ms per query on a T4.

## Repository contents
- `VizWiz_Fusion.ipynb`: full pipeline (data checks, features, models, evaluation, CLIP)
- `README.md`: this file

## Ethics
Only code is shared, with no images. Results are reported by image sharpness and
question frequency to check bias. The model is meant to prompt a retake, not to replace
human help, especially for medication or safety questions.
