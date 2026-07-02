# Pneumonia detection from chest X-Rays using VGG16

We are building a deep learning system to detect pneumonia from chest X-ray images. Using a Convolutional Neural Network (CNN) based on the VGG16 architecture to improve diagnostic accuracy.This project processes raw medical imaging data (DICOM) and applies transfer learning to classify scans into `Pneumonia` and `Not Pneumonia` categories.

## Installation

Follow these instructions to set up a local copy of the project and deploy the execution environment.

### Prerequisites

The pipeline requires Python 3.8+ along with deep learning, medical imaging, and data science libraries. Since the project is structured around Jupyter Notebooks, ensure Jupyter is installed.

Install the required dependencies using pip:

    pip install tensorflow pydicom opencv-python scikit-learn matplotlib numpy jupyter

## Usage

To execute the pipeline and train the model, follow the steps below:

1. **Dataset Acquisition:** Visit the official [RSNA Pneumonia Detection Challenge 2018](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) page. You will need to download the files using the following two specific links provided on their site:
    * **"Download images from NIH chest x-ray dataset used in the Pneumonia Challenge"**
    * **"Download annotations used in the Pneumonia Challenge"**

2. **Data directory setup:** Create a folder named `raw_data` in the root directory of the project (next to the notebooks). Extract the downloaded dataset so it matches the following structure:
```text
.
├── data_load.ipynb
├── vgg16_model.ipynb
└── raw_data/
    ├── pneumonia-challenge-annotations-adjudicated-kaggle_2018.json
    └── pneumonia-challenge-dataset-adjudicated-kaggle_2018/
        └── mdai_rsna_project_x9N20BZa_images_2018-07-20-153330/
            ├── 1.2.276.0.7230010.3.1.2.8323329.1472.1517874291.114974/
                └── 1.2.276.0.7230010.3.1.3.8323329.1472.1517874291.114973/
                    └── 1.2.276.0.7230010.3.1.4.8323329.1472.1517874291.114975.dcm
            ├── 1.2.276.0.7230010.3.1.2.8323329.1473.1517874291.110144/
            └── ...
```
  

The `data_load.ipynb` notebook is pre-configured with these relative paths by default.

3. **Execution:** 
    Execute the notebooks strictly in the following order
    
    * **Step 1 - Run `data_load.ipynb`:** Execute all cells. This notebook parses the JSON annotations, maps the file identifiers, splits the dataset (train/val/test), and copies the raw DICOM files into a structured `data/` directory.
    * **Step 2 - Run `vgg16_model.ipynb`:** Execute all cells. This notebook loads the custom generators from the generated `data/` folder, builds the VGG16 architecture, trains the model, outputs the final evaluation metrics.

## Deploy

The project incorporates automatic model persistence during and after the training sequence within `vgg16_model.ipynb`:

* **Intermediate checkpoints:** The `ModelCheckpoint` callback monitors `val_loss` and automatically serializes the best performing weights to `models/forsafety.keras`.
* **Final model serialization:** Upon completion or early termination via `EarlyStopping`, the fully trained model architecture and its weights are saved to `models/pneumonia_vgg16.h5` for downstream inference or deployment.

## Technologies

The following technologies and frameworks were utilized to build this project:

* [Python](https://www.python.org/) - Core programming language.
* [Jupyter](https://jupyter.org/) - Interactive computational environment.
* [TensorFlow / Keras](https://www.tensorflow.org/) - Deep learning framework and model architecture API.
* [Pydicom](https://pydicom.github.io/) - Medical image parsing and metadata extraction library.
* [OpenCV](https://opencv.org/) - Digital image processing, resizing, and color space transformations.
* [Scikit-learn](https://scikit-learn.org/) - Dataset stratification and validation metric utilities.
* [Matplotlib](https://matplotlib.org/) - Metrics visualization and confusion matrix plotting.

## Documentation

### 1. Data processing pipeline (`data_load.ipynb`)
Medical datasets require specialized indexing. This module addresses several preprocessing challenges:
* **Identifier resolution:** In the annotation JSON, the labels are linked to StudyInstanceUID, but the actual DICOM files are named by SOPInstanceUID. The script pairs these up so every label finds its correct file, avoiding file-not-found errors.
* **Stratified dataset splitting:** Automatically splits the filtered dataset into training (70%), validation (15%), and test (15%) subsets. It uses stratified splitting to keep the ratio of pneumonia and healthy cases equal across all three sets.
* **File management:** Automatically constructs the target directory tree and copies the respective files, preparing an isolated environment for the training generators.

### 2. Model architecture and training (`vgg16_model.ipynb`)
This module handles data loading, model compilation, and the training loop.
* **Custom memory-efficient loader:** Implements a `DicomDataGenerator` inheriting from `keras.utils.Sequence`. It handles pixel value normalization (0-255 scale), resizing to 224x224x3, and real-time data augmentation (Gaussian blur and random noise injection) only on the training split.
* **Feature extractor:** The network is built upon the VGG16 model pre-trained on ImageNet. The initial convolutional blocks are frozen, while the final 4 layers are set to trainable.
* **Classifier head:** Features a customized top network including `GlobalAveragePooling2D`, a dense network structure (256 -> 128 -> 64 units) with `ReLU` activations, and `Dropout` layers (0.5 and 0.3) to prevent overfitting. 
* **Optimization:** Employs the Adam optimizer with a conservative learning rate. Class weights are computed dynamically to balance the loss function against class distribution asymmetry.

### 3. Evaluation and results
Training curves (loss and accuracy) are plotted after training to monitor model convergence and to easily identify potential underfitting or overfitting. The final performance is measured on a separate test set that the model has never seen before.
<img width="1189" height="490" alt="val_train_curves" src="https://github.com/user-attachments/assets/97c33e8c-fa53-4173-908e-30cb1a38bda3" />

Due to random weight initialization and the localized training splits, the test accuracy typically varies between 78% and 82%.

The system also includes a visual testing tool. It processes individual X-ray images, makes a prediction, then displays the raw image alongside the model's confidence score. If the prediction is correct, the text is highlighted in green, while mistakes are marked in red. Using 0.5 for decision treshold.

<table>
  <tr>
    <td><img width="200" alt="xrr_4" src="https://github.com/user-attachments/assets/7b77d234-8feb-4b12-ad25-eb0563f99457" /></td>
    <td><img width="200" alt="xrr_3" src="https://github.com/user-attachments/assets/ac937d6c-b6c7-4df6-a93a-8d27d0305523" /></td>
    <td><img width="200" alt="xrr_1" src="https://github.com/user-attachments/assets/dd0f2efb-cede-48d3-8e1b-b78eaa7b3674" /></td>
    <td><img width="200" alt="xrr_2" src="https://github.com/user-attachments/assets/e45babad-2d77-475d-8d8c-5b33f7620b69" /></td>
  </tr>
</table>

To optimize the diagnostic threshold, the evaluation module generates classification reports alongside visual confusion matrices across multiple decision thresholds (0.5, 0.4, 0.3, 0.2). This allows for a detailed analysis of the trade-off between sensitivity (recall) and specificity, ensuring a safer clinical application.
<img width="2312" height="490" alt="conf_matrixes" src="https://github.com/user-attachments/assets/6dd2c592-173f-42d7-aaec-bed2d8ba5e3b" />

## Acknowledgments

* [RSNA Pneumonia Detection Challenge 2018](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018) - Dataset and challenge providers.
* National Institutes of Health (NIH) Clinical Center for providing the underlying dataset.
* Kaggle platform for hosting the adjudicated evaluation metadata.
