# Pneumonia detection using CNN in Python
We are building a deep learning system to detect pneumonia from chest X-ray images. Using a Convolutional Neural Network (CNN) based on the VGG16 architecture to improve diagnostic accuracy.This project processes raw medical imaging data (DICOM) and applies transfer learning to classify scans into `Pneumonia` and `Not Pneumonia` categories.

## Data processing pipeline

The preprocessing module performs the following operations:

* **Annotation parsing:** Reads clinical annotations from Kaggle's RSNA Pneumonia dataset (JSON), mapping `StudyInstanceUID` to physical `SOPInstanceUID` file identifiers.
* **Automated data organization:** Automatically splits the dataset (70% train, 15% validation, 15% test) using stratified sampling and copies the raw `.dcm` files into a clean, hierarchical directory structure.
* **Custom data generator:** Implements a memory-efficient `DicomDataGenerator` utilizing Keras `Sequence`. It dynamically loads, normalizes, and resizes DICOM images to 224x224 pixels in batches.
* **Data augmentation:** Applies random Gaussian blur and noise injection during training to improve model generalization.

## Model architecture

The classification model relies on transfer learning using the **VGG16** network architecture pre-trained on ImageNet.

* **Feature extractor:** VGG16 base model with the last 4 layers unfrozen for fine-tuning.
* **Custom classifier head:** * `GlobalAveragePooling2D` for dimensionality reduction.
  * Deep Dense network layers (256 -> 128 -> 64) with `ReLU` activations.
  * `BatchNormalization` and `Dropout` (0.5 and 0.3) layers to prevent overfitting.
  * Final `Dense` layer with `Sigmoid` activation for binary classification.
* **Optimization:** `Adam` optimizer with dynamically adjusted learning rates (`ReduceLROnPlateau`), `EarlyStopping`, and `ModelCheckpoint` to save the best weights. Class weights are applied to handle dataset imbalance.

## Evaluation and results
The training progress is monitored through loss and accuracy curves plotted across all epochs, allowing for the analysis of convergence and potential overfitting. The model's final performance is measured on a strictly unseen test set.

- **Training/validation plots:** Loss and accuracy curves are generated post-training to visualize the learning trajectory and model stability.
- **Classification metrics:** A comprehensive classification report (Precision, Recall, F1-Score) is computed on the test set.
- **Confusion matrices:** Evaluation is performed across multiple probability thresholds (0.5, 0.4, 0.3, 0.2) to analyze the trade-off between sensitivity and specificity.

Due to the statistical nature of deep learning and random initializations, the **Test accuracy currently fluctuates between 78% and 81%**.

## Setup and execution

1. **Dataset acquisition:**
   Download the required images and annotation files from the official [RSNA Pneumonia Detection Challenge 2018](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) website. Ensure both the image directory and the annotation JSON file are downloaded and extracted locally.
2. **Environment setup:**