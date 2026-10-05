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

_To fill in after the runs. Keep the failures in._

| Trained on | Tested on | AUROC (95% CI) | Sensitivity | Specificity |
|---|---|---|---|---|
| Shenzhen | Shenzhen (test split) | | | |
| Shenzhen | Montgomery (all) | | | |
| Montgomery | Montgomery (test split) | | | |
| Montgomery | Shenzhen (all) | | | |

## Limitations (write these honestly)

- Both sets are small (hundreds of images), so confidence intervals are wide.
- Two sources is a minimal definition of "external". Neither set represents
  Nepal or South Asia.
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
committed.

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

Cite the dataset papers (Jaeger et al. 2014 for Montgomery and Shenzhen) and
the ResNet and Grad-CAM papers when you write this up.
