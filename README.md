# CGAN-MRI
Accelerated Magnetic Resonance Imaging (MRI) seeks to reduce scan times by acquiring undersampled k-space data, resulting in an ill-posed inverse problem that degrades image quality. This project reproduces and evaluates a conditional generative adversarial network (CGAN) for cardiac MRI reconstruction, based on the paper *"A Deep Learning Framework for Cardiac MR Under-Sampled Image Reconstruction with a Hybrid Spatial and k-Space Loss Function"* by Al-Haidri et al. (2023). Using the OCMR dataset, a reproducible procedure was implemented to create undersampled inputs and corresponding ground truth images. The model was trained and evaluated across acceleration factors of R=2, R=4 and R=8, with performance assessed using SSIM and PSNR. The reproduction achieves strong reconstruction quality at lower acceleration factors, with effective suppression of aliasing artifacts and preservation of structural detail. Notably, the proposed implementation outperforms the original study while using approximately one quarter of the parameters, demonstrating that competitive reconstruction quality can be achieved with significantly greater parameter efficiency. The project also highlights the reproducibility challenges that arise from incomplete architectural specification in the source paper.

---

## Project Structure

```
CGAN-MRI/
├── callbacks/
├── data/
├── dataset_generation/
├── losses/
├── model/
├── pl_modules/
├── train.py
├── eval.py
├── eval-2.py
├── inference.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Requirements

### Installation

Clone the repository:
```bash
git clone https://github.com/nikhilsantokhi/CGAN-MRI.git
cd CGAN-MRI
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Additional Dependency: read_ocmr

This project uses `read_ocmr` for loading OCMR data, which is not available via pip. You must download it manually from the OCMR GitHub repository:

1. Visit https://github.com/MRIOSU/OCMR
2. Download `read_ocmr.py`
3. Place it in the `dataset_generation/` folder

This file must be present before running any dataset generation scripts.

---

## Dataset

This project uses the [OCMR dataset](https://www.ocmr.info/)
(Open-Access Multi-Coil k-Space Dataset for Cardiovascular MRI).

1. Download the fully sampled scans from https://ocmr.info/download/
2. Place the `.h5` files in the appropriate directory as expected by `dataset_generation/`

Two scripts must be run in order to generate the dataset:

**Step 1** — Generate aliased and ground truth image pairs:
```bash
python dataset_generation/generate_dataset.py
```

**Step 2** — Split the dataset into train, validation and test sets:
```bash
python dataset_generation/split_dataset.py
```

Before running, open each script and set the relevant directory paths.

---

## Training

Before training, open `train.py` and set the following variables:
- Data directory paths
- Acceleration factor (R=2, R=4, or R=8)


Training logs and reconstructed images are monitored via TensorBoard:

```bash
tensorboard --logdir checkpoints/
```

---

## Inference

Before running evaluation, inference must be run first to generate model outputs. Open `inference.py` and set the following:
- Data directory paths
- Path for the saved  model
- Acceleration factor (R=2, R=4, or R=8)

Then run:

```bash
python inference.py
```

The generated outputs will be saved and used by the evaluation scripts.

---

## Evaluation

Open `eval.py` and set the paths to the inference outputs, then run:

```bash
python eval.py
```

This will compute SSIM and PSNR metrics on the test set.
## Evaluation

Similarly, open `eval.py` and set the data and checkpoint paths, then run:

```bash
python eval.py
```

This will compute SSIM and PSNR metrics on the test set.

---

## Results

| Acceleration Factor | SSIM | PSNR (dB) |
|---|---|---|
| R=2 | 0.9404 | 38.91 |
| R=4 | 0.9212 | 36.98 |
| R=8 | 0.8326 | 31.55 |

---

## References

Walid Al-Haidri et al. *“A Deep Learning Framework for Cardiac MR Under-Sampled Image Reconstruction with a Hybrid Spatial and k-Space Loss Function”*. In: Diagnostics 13.6 (Mar. 2023). issn: 20754418. doi: 10.3390/diagnostics13061120.

Chong Chen et al. *“OCMR (v1.0)–Open-Access Multi-Coil k-Space Dataset for Cardiovascular Magnetic Resonance Imaging”*. In: (Aug. 2020). url: http://arxiv.org/abs/2008.03410. 

