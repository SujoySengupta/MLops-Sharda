```markdown
# Simple ML Training Pipeline

This repository contains a basic machine learning training pipeline. The script demonstrates how to load a dataset, split it into training and testing sets, train a Logistic Regression classifier, evaluate its performance, and serialize the trained model for future deployment.

## Features

* **Data Handling:** Loads the Breast Cancer dataset using `scikit-learn` and processes it with `pandas`.
* **Model Training:** Trains a Logistic Regression model with optimized iteration limits to ensure convergence.
* **Evaluation:** Outputs the model's accuracy score alongside a comprehensive classification report (precision, recall, f1-score).
* **Serialization:** Saves the trained model artifact as a `.pkl` file using `joblib` for efficient storage and deployment.

## Project Structure

```text
├── train_model.py              # Main script for training and saving the model
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── logistic_regression_model.pkl # Generated model artifact (after running the script)

```

## Prerequisites

* Python 3.8 or higher

## Setup and Installation

It is recommended to use a virtual environment to manage dependencies.

1. **Create a virtual environment:**
```bash
python -m venv venv

```


2. **Activate the virtual environment:**
* **Windows:**
```bash
venv\Scripts\activate

```


* **macOS/Linux:**
```bash
source venv/bin/activate

```




3. **Install the required packages:**
```bash
pip install -r requirements.txt

```



## Usage

Run the training script from your terminal:

```bash
python train_model.py

```

### Expected Output

When you run the script, it will print the progress to the console, followed by the evaluation metrics and a confirmation that the model has been saved:

```text
Loading data...
Splitting data...
Training model...

Evaluating model...
------------------------------
Model Accuracy: 95.61%
------------------------------
Classification Report:
... (precision, recall, f1-score details) ...

Model saved successfully as 'logistic_regression_model.pkl'

```
