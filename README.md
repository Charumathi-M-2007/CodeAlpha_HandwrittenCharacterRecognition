# Handwritten Digit Recognition Using Machine Learning

## Overview

Handwritten Digit Recognition is a machine learning project that identifies handwritten numerical digits from image data. It uses a **Random Forest Classifier** to classify digits from 0 to 9 based on their pixel values.

The project demonstrates the complete machine learning workflow, including dataset loading, data preprocessing, model training, prediction, performance evaluation, and result visualization.

## Objectives

- Develop a machine learning model to recognize handwritten digits.
- Train and test the model using labeled image data.
- Evaluate classification performance using standard metrics.
- Visualize predicted digits and model performance.
- Save the trained model for future use.

## Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Data Processing:** NumPy
- **Data Visualization:** Matplotlib
- **Model Serialization:** Joblib

## Algorithm

**Random Forest Classifier**

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve classification performance and reduce the risk of overfitting.

## Dataset

This project uses the built-in handwritten digits dataset provided by Scikit-learn.

- **Classes:** Digits 0–9
- **Image dimensions:** 8 × 8 pixels
- **Features:** 64 pixel-intensity values per image
- **Data type:** Labeled grayscale digit images

The dataset is divided into training and testing sets to evaluate the model on unseen examples.

## Project Structure

```text
HandwrittenCharacterRecognition/
├── handwritten_recognition.py
├── requirements.txt
├── README.md
├── handwritten_model.pkl
├── sample_predictions.png
└── confusion_matrix.png
```

*The model and image files are generated when the Python script runs successfully.*

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Charumathi-M-2007/CodeAlpha_HandwrittenCharacterRecognition.git
```

### 2. Navigate to the Project Directory

```bash
cd CodeAlpha_HandwrittenCharacterRecognition
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Execution

Run the following command:

```bash
python handwritten_recognition.py
```

The program trains the classifier, evaluates its performance, displays sample predictions, and generates a confusion matrix.

## Model Evaluation

The model's performance is evaluated using:

- **Accuracy:** Measures the proportion of correctly classified digits.
- **Precision:** Measures the correctness of positive predictions for each class.
- **Recall:** Measures how many actual examples of each class are correctly identified.
- **F1-Score:** Balances precision and recall.
- **Confusion Matrix:** Shows correct and incorrect classifications across digit classes.

The actual performance metrics are displayed when the program is executed.

## Output

The program generates:

- Model accuracy and classification report.
- Sample images showing actual and predicted digits.
- A confusion matrix for classification analysis.
- A saved trained model named `handwritten_model.pkl`.

## Limitations

- The current implementation recognizes numerical digits from 0 to 9 only.
- It uses small 8 × 8 pixel images.
- It is not designed to recognize arbitrary uploaded images or alphabetic characters directly.

## Future Enhancements

- Extend recognition to handwritten letters A–Z.
- Support user-uploaded handwritten images.
- Develop a web interface for interactive predictions.
- Explore Convolutional Neural Networks (CNNs) for image recognition.

## Learning Outcomes

This project provides practical experience in machine learning classification, dataset preparation, model evaluation, image visualization, and saving trained models.

## Disclaimer

This project is intended for educational and demonstration purposes. Performance on the built-in dataset does not guarantee equivalent results on real-world handwritten images.

