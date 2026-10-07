# TB chest X-ray screening: does it carry over to another hospital?

I trained a ResNet-18 to spot tuberculosis on chest X-rays from one public
dataset, then tested it on a dataset from a different hospital to see how much
it drops. I care more about that drop than about a high score.

This is a research demo. It is not a medical tool and should not be used to
diagnose anything.

## What I did

The data is two public sets from the US National Library of Medicine:
Montgomery County (USA, 138 images) and Shenzhen (China, 662 images). Each file
is a different patient, so splitting by file also splits by patient.

The model is an ImageNet-pretrained ResNet-18 with its last layer swapped for
a single output. The loss is weighted for class imbalance, and I used light
augmentation (small rotations and brightness changes, no flips).

For each direction (train on A, test on B, then swap):

1. Split A into train, validation and test (about 70/15/15).
2. Train for 15 epochs and keep the epoch with the best validation AUROC.
3. Pick the decision threshold on the validation split, at 90% specificity.
   The test data never touches this.
4. Test on A's own test split and on every image of B.

I report AUROC, sensitivity and specificity instead of accuracy, because in
screening a missed case and a false alarm cost different things.

Each direction was run with 5 seeds. The seed changes both the split and the
initial weights, so the spread below includes both.

## Results

Mean ± standard deviation over 5 seeds.

| Trained on | Tested on | Images (TB) | AUROC | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Shenzhen | Shenzhen test split | 100 (51) | 0.953 ± 0.018 | 0.78 ± 0.10 | 0.93 ± 0.04 |
| Shenzhen | Montgomery, all | 138 (58) | 0.841 ± 0.033 | 0.41 ± 0.12 | 0.95 ± 0.03 |
| Montgomery | Montgomery test split | 21 (9) | 0.948 ± 0.018 | 0.62 ± 0.22 | 0.98 ± 0.04 |
| Montgomery | Shenzhen, all | 662 (336) | 0.806 ± 0.029 | 0.52 ± 0.17 | 0.86 ± 0.15 |

The ranking gets worse on the other hospital's images. In all 10 runs the
external AUROC was lower than the in-domain one, going from about 0.95 to about
0.81 to 0.84. The size of the drop varies a lot between seeds, from 0.03 up to
0.19.

## What did not work

The threshold. It was meant to give 90% specificity, but on the other
hospital's images specificity ended up anywhere from 59% to 99%, and
sensitivity from 21% to 76%. The thresholds themselves were all over the place
(0.81 to 0.99 for the Shenzhen runs, 0.16 to 1.0 for the Montgomery runs). Two
Montgomery runs hit a validation AUROC of 1.0 on a validation split of about
20 images, so the threshold they chose was close to arbitrary.

My guess is that it comes down to overfitting (training loss goes to nearly
zero) plus very small validation sets. I did not test that.

Used as a screening tool at its fixed threshold, the Shenzhen model would miss
more than half of the TB cases in the Montgomery set, on average.

The Montgomery test split has only 21 images (9 TB), so the in-domain rows for
that model mean very little.

## Grad-CAM

I made heatmaps for 12 external images (3 TB and 3 normal per model, the first
ones in file order, not picked). They came from the first seed's models. I'm
not including them here because they are the original X-rays with a heatmap
on top, and the Shenzhen terms say not to share the images.

For the Shenzhen model on Montgomery, the TB cases lit up on lung tissue. For
normal cases the heat sat in the lower corner near the heart and diaphragm.
For the Montgomery model on Shenzhen, two TB cases were missed, and in one of
them the only heat was over the diaphragm. One normal case had its heat in the
top corner next to the "L" marker, outside the lungs. This hints at the model
using things that are not lungs, but twelve images cannot show that.

## Limitations

- Both sets are small, so everything is noisy. The validation splits are about
  20 (Montgomery) and 100 (Shenzhen) images.
- The two sets differ in country, scanner, population and image format at the
  same time. I can't say which of those causes the drop.
- Two sources is a minimal "external" test. Neither set covers South Asia.
- The labels come from the dataset curators. I did not check them.
- Doing well on these images says nothing about doing well in a clinic.

## Getting the data

I did not include the images. Download both sets from the NLM page "Tuberculosis
Chest X-ray Image Data Sets" and check their terms. The Shenzhen read-me asks
that the data not be shared outside your research group. Put the PNGs here:

```
data/montgomery/MCUCXR_0001_0.png ...
data/shenzhen/CHNCXR_0001_1.png ...
```

The last digit in the file name is the label (0 normal, 1 TB). If the data
lives somewhere else, pass `--data-root <folder>` to the commands below.

## Running it

```bash
pip install -r requirements.txt
python -m src.train --train-on shenzhen --epochs 15
python -m src.evaluate --train-on shenzhen
python -m src.train --train-on montgomery --epochs 15
python -m src.evaluate --train-on montgomery
python -m src.gradcam --train-on shenzhen --images data/montgomery/MCUCXR_0001_0.png
```

Extra seeds go in their own folder so they don't overwrite the first run:

```bash
python -m src.train --train-on shenzhen --seed 1 --run-dir results/seeds/shenzhen_seed1
python -m src.evaluate --train-on shenzhen --run-dir results/seeds/shenzhen_seed1
python -m src.aggregate --train-on shenzhen --runs results/shenzhen results/seeds/shenzhen_seed1
```

Reports are written to `results/<train_on>/report.json`. Tests: `pytest`.

## References

Data: National Library of Medicine, NIH, Bethesda, MD, USA; Shenzhen No.3
People's Hospital, Guangdong Medical College, Shenzhen, China; Montgomery County
Department of Health and Human Services, Maryland, USA.

- Jaeger S, et al. Automatic tuberculosis screening using chest radiographs.
  IEEE Trans Med Imaging. 2014;33(2):233-245.
- Candemir S, et al. Lung segmentation in chest radiographs using anatomical
  atlases with nonrigid registration. IEEE Trans Med Imaging. 2014;33(2):577-590.
- He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition.
  CVPR 2016.
- Selvaraju RR, et al. Grad-CAM: visual explanations from deep networks via
  gradient-based localization. ICCV 2017.
