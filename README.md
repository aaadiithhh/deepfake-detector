# Deepfake Face Detector

This is a face classifier made with PyTorch. It outputs either real or fake for each image.

## What the project does

I start from a pretrained ResNet18. Then I train it to tell apart real photos from synthetic faces. The fake side comes from GANs. This work is meant as a practical deep learning exercise. It goes from dataset setup to a script that can predict.

## Data

- Source: Kaggle dataset “140k Real and Fake Faces”  
  Link: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
- Images used: 5,000 for training and 1,000 for validation  
  These are only part of the full set.
- Labels:  
  real means authentic photographs  
  fake means GAN-made synthetic faces

## How it was built

1. Preprocessing  
   Each image is resized to 224 by 224. Then it is normalized with ImageNet stats.
2. Model  
   ResNet18 pretrained on ImageNet. I swap the last layer so it predicts 2 classes. So it is real vs fake, not 1,000 categories.
3. Training  
   I fine-tune for 3 epochs. Adam is used. The loss is cross-entropy.
4. Evaluation  
   After every epoch, I log training and validation accuracy. This helps check how well the model generalizes.

## Outcomes

| Epoch | Train Acc | Val Acc |
|-------|----------:|---------:|
| 1     | 85.48%    | 81.00%   |
| 2     | 97.86%    | 93.30%   |
| 3     | 99.44%    | 91.60%   |

The best validation score is at epoch 2, with 93.30%. After that, validation drops, even though training keeps improving. That pattern points to mild overfitting in epoch 3, when the model starts fitting the training images more closely instead of learning signals that transfer.

## Key Limitation

This model is trained specifically on **GAN-generated synthetic faces** (entirely AI-created images). It has not been trained on **manipulated real photos** (e.g., face swaps or diffusion-based edits of an existing photo), and testing confirmed it does not reliably detect that second category. These are visually and statistically distinct forms of "fake" images, and a production-ready detector would need training data covering both.

## Files

- `train.py` — loads the dataset, fine-tunes the model, and saves the trained weights
- `predict.py` — loads a saved model and predicts real/fake on a given image
- `deepfake_detector.pth` — trained model weights

## How to Run

```bash
# Train the model
python train.py

# Test on an image
python predict.py
```

## Tech Stack

- Python
- PyTorch / Torchvision
- ResNet18 (transfer learning)

## Future Improvements

- Add data augmentation (random flips/crops) and dropout to reduce overfitting
- Train on a larger portion of the dataset for more epochs
- Incorporate a second dataset (e.g., FaceForensics++) covering face-swap and edited-photo manipulations
- Wrap the model in a web interface (Flask) and/or desktop app for interactive use
