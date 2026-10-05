# TB chest X-ray screening: does it generalise?

A small, honest study of deep-learning tuberculosis screening from chest X-rays.
The point is not a high accuracy number. It is measuring **how much a model's
performance drops when it meets images from a different hospital**, and looking
at why.

> **Research demo only. Not a medical device and not for diagnosis.**

## Question

A ResNet-18 trained on one public TB dataset scores very well on a held-out
slice of that same dataset. How does it do on a dataset from a different
country, scanner and population, with the decision threshold fixed in advance?

## Method

- **Data:** two public sets from the US National Library of Medicine, the
  Montgomery County set (USA) and the Shenzhen set (China). One image per patient,
  so splitting by file is a patient-level split.
- **Model:** ImageNet-pretrained ResNet-18, single-logit head, class-weighted
  loss, mild augmentation (no flips).
- **Protocol:** train on A with a stratified train/val/test split. Pick the
  threshold on A's **validation** set at 90% specificity. Evaluate on A's test
  split (in-domain) and on **all** of B (external). Repeat with A and B swapped.
- **Metrics:** AUROC with bootstrap 95% CIs, sensitivity and specificity at
  the fixed threshold. Accuracy is deliberately not the headline: for screening,
  missed cases and false alarms cost different things.
- **Interpretation:** Grad-CAM heatmaps to check whether the model looks at the
  lungs or at artefacts such as borders and markers.

## Results

One run per direction (seed 0, one split, 15 epochs, best epoch by validation
AUROC). The threshold was fixed on the training set's validation split at 90%
specificity and never changed. Sensitivity and specificity come with counts
because the sets are small.

| Trained on | Tested on | n (TB) | AUROC (95% CI) | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Shenzhen | Shenzhen (test split) | 100 (51) | 0.967 (0.933-0.990) | 0.667 (34/51) | 0.980 (48/49) |
| Shenzhen | Montgomery (all) | 138 (58) | 0.826 (0.757-0.889) | 0.414 (24/58) | 0.950 (76/80) |
| Montgomery | Montgomery (test split) | 21 (9) | 0.954 (0.850-1.000) | 0.333 (3/9) | 1.000 (12/12) |
| Montgomery | Shenzhen (all) | 662 (336) | 0.810 (0.778-0.841) | 0.336 (113/336) | 0.957 (312/326) |

Fixed thresholds: 0.978 (Shenzhen model), 0.962 (Montgomery model).

### What the numbers say

- **Ranking drops across hospitals.** AUROC falls from about 0.95-0.97 in-domain
  to about 0.81-0.83 on the other set, in both directions. The in-domain and
  external confidence intervals do not overlap for the Shenzhen-trained model;
  for the Montgomery-trained model they only just separate (0.850 vs 0.841),
  because its in-domain test split has 21 images.
- **The 90%-specificity threshold did not hold up.** Specificity landed at
  95-100% instead of 90%, and sensitivity at the fixed threshold is poor:
  34-41% on external data, and only 67% (Shenzhen) and 3 of 9 (Montgomery)
  in-domain. The models are overconfident after fitting the training set
  (train loss near 0.01-0.05), so a threshold picked on a small validation
  split sits too high. As a screening tool, either model would miss most TB
  cases on the other hospital's images.
- **Montgomery in-domain is nearly meaningless.** Its test split has 21 images
  (9 TB); treat those two rows as anecdotes.

### Grad-CAM (qualitative, 12 images)

Heatmaps for the first 3 TB and first 3 normal external images per model were
generated with `src.gradcam`. They are not committed, because each one is the
source X-ray with a heatmap on top and the Shenzhen terms ask that the images
not be shared. They are not cherry-picked, but 12 images cannot prove anything.

- Shenzhen model on Montgomery: heat on lung tissue for the TB cases; for
  normal cases it concentrates in the lower corner near the heart and diaphragm.
- Montgomery model on Shenzhen: one TB case highlights the upper lung, two TB
  cases are missed (P(TB) 0.01 and 0.00), and one of those has its only heat
  over the diaphragm/abdomen. One normal case has its heat in the top corner
  beside the "L" marker, outside the lungs.
- Maps are rescaled to 0-1 per image, so for low-probability images they show
  where the model is relatively most active, not evidence of disease. The
  off-lung heat is a hint of non-lung cues, not a demonstrated shortcut.

## Limitations

- Both sets are small (hundreds of images), so confidence intervals are wide.
  The validation splits are only about 20 (Montgomery) and 100 (Shenzhen)
  images, which makes the chosen epoch and threshold noisy.
- Single seed and a single split per direction; no repeats, so run-to-run
  variance is unmeasured. No confidence intervals on sensitivity/specificity.
- Two sources is a minimal definition of "external". Neither set represents
  Nepal or South Asia.
- The two sets differ in country, scanner, population and image format at once;
  this study cannot say which of those causes the drop.
- Labels come from the dataset curators, not an independent reading here.
- A model that works on these images is not evidence it works in a clinic.

## Getting the data

Download the two sets manually (check each set's licence and terms) and place
the PNGs like this:

```
data/montgomery/MCUCXR_0001_0.png ...
data/shenzhen/CHNCXR_0001_1.png ...
```

The final digit in each file name is the label (0 = normal, 1 = TB). Search for
"Tuberculosis Chest X-ray Image Data Sets" from the U.S. National Library of
Medicine (Montgomery County and Shenzhen). Data is gitignored and must not be
committed. The Shenzhen set's read-me asks that the data not be shared outside
your research group.

If the data lives elsewhere, pass `--data-root <folder>` to `src.train` and
`src.evaluate` (the folder holds `montgomery/` and `shenzhen/`).

## Run

```bash
pip install -r requirements.txt
python -m src.train --train-on shenzhen --epochs 15
python -m src.evaluate --train-on shenzhen
python -m src.train --train-on montgomery --epochs 15
python -m src.evaluate --train-on montgomery
python -m src.gradcam --train-on shenzhen --images data/montgomery/MCUCXR_0001_0.png
```

Reports land in `results/<train_on>/report.json`. Tests: `pytest`.

## References

Data: National Library of Medicine, NIH, Bethesda, MD, USA; Shenzhen No.3
People's Hospital, Guangdong Medical College, Shenzhen, China; Montgomery County
Department of Health and Human Services, Maryland, USA.

- Jaeger S, et al. Automatic tuberculosis screening using chest radiographs.
  IEEE Trans Med Imaging. 2014;33(2):233-245.
- Candemir S, et al. Lung segmentation in chest radiographs using anatomical
  atlases with nonrigid registration. IEEE Trans Med Imaging. 2014;33(2):577-590.

Also cite the ResNet and Grad-CAM papers when you write this up.
