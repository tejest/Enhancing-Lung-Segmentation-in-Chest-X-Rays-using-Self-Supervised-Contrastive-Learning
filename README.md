This project explores the use of Self-Supervised Learning (SSL) techniques to improve lung segmentation performance on Chest X-Ray images.

The approach leverages pretext tasks such as:

- Rotation Prediction
- SimCLR Contrastive Learning
- MoCo Contrastive Learning

The learned representations are transferred to a supervised segmentation network and evaluated on public lung segmentation datasets.

## MOTIVATION":
Accurate lung segmentation is critical for:

- Pneumonia detection
- Tuberculosis screening
- COVID-19 analysis
- Computer-Aided Diagnosis systems

Manual annotation is expensive and time-consuming.

Self-Supervised Learning helps learn robust visual representations without requiring large amounts of labeled data.


## TRAINING PIPELINE

1. Self-Supervised Pretraining
   - Rotation Prediction
   - SimCLR
   - MoCo

2. Encoder Weight Transfer

3. Supervised Segmentation Training

4. Evaluation on JSRT and Montgomery datasets

Datasets

Mention:

## Datasets

JSRT Dataset
- Chest X-Ray images
- Lung masks

Montgomery Dataset
- Tuberculosis screening X-rays
- Expert annotated lung masks

## Installation
git clone https://github.com/yourusername/project.git

cd project

pip install -r requirements.txt

## Training
python train_pretext_rotation.py

python train_contrastive_simclr.py

python train_contrastive_moco.py

python train_supervised_cv.py
Prediction
python predict_cxr.py --image sample.png

## ARCHITECTURE
Chest X-Ray
     │
     ▼
Self-Supervised Pretraining
     │
 ┌───┼────┐
 ▼   ▼    ▼
Rot SimCLR MoCo
     │
     ▼
Encoder
     │
     ▼
Segmentation Network
     │
     ▼
Lung Masks

## Key Highlights

✅ Self-Supervised Learning

✅ Medical Image Segmentation

✅ Contrastive Learning

✅ PyTorch

✅ Computer Vision

✅ Transfer Learning

✅ Representation Learning
