# Parkinson-EEG-Detection
A deep learning and machine learning-based approach for the early detection of Parkinson’s Disease using EEG signal analysis.

## Title of the project: 
### Leveraging EEG Signal Analysis and Machine Learning for Early Detection of Parkinson's Disease: A Comprehensive Approach

## Project Overview
Parkinson's Disease (PD) is a chronic, progressive neurological disorder affecting millions globally. Early detection is crucial for effective management and treatment. This project leverages EEG signal processing and AI-based models to identify PD at its early stages.

Our goal is to develop a robust and accurate system that processes EEG signals and classifies individuals as healthy or affected by Parkinson’s, using a combination of machine learning and deep learning models.

## Objectives
* Extract and preprocess EEG signal data for analysis.
* Implement multiple machine learning models (e.g., Logistic Regression, PCA).
* Build and train deep learning models (ANN and CNN).
* Compare performance and accuracy across models.
* Deploy the best-performing model through a web interface.
* Enable real-time EEG classification for clinical and research use.

## Technologies Used
* Python 3.x
* Pandas, NumPy
* Scikit-learn, TensorFlow, Keras
* Matplotlib, Seaborn
* Streamlit (Web Application)
* EEG Dataset (external)

## Models Implemented

| Model                        | Role                                           |  Accuracy                  |     
|------------------------------|----------------------------------------------- |----------------------------|
| Logistic Regression	         | Baseline machine learning model	              | Moderate                   |
| PCA + ML Classifier          |Dimensionality reduction + classification	      | Good                       |
| Artificial Neural Network	   | Deep learning model capturing complex patterns |	High                       |
| Convolutional Neural Network |	Best performing DL model for spatial features	| Best (100%)                |

## Dataset
The EEG dataset includes readings from both healthy individuals and Parkinson's patients. Preprocessing steps include:

* Signal denoising and normalization
* Feature extraction
* Data segmentation
* Label encoding

##  Project Structure
ParkinsonAI-EEG-Detection/
data/                   # EEG dataset
models/                 # Saved ML/DL models
app/                    # Streamlit application
preprocessing/          # Scripts for data cleaning
main.py                 # Model training/testing
app.py                  # Streamlit app main file
requirements.txt
README.md

## Getting Started
### Clone the repository
```
git clone https://github.com/praisysera/ParkinsonAI-EEG-Detection.git
cd ParkinsonAI-EEG-Detection
```

### Install dependencies
```
pip install -r requirements.txt
```
### Run the Streamlit app
```
streamlit run app.py
```

### Streamlit Web App Features
* Upload EEG signal data
* Instant classification using trained CNN
* Displays model accuracy and predictions
* Clean, responsive UI for researchers and clinicians

### Future Scope
* Integration with real-time EEG acquisition devices
* Enhancement with larger and more diverse datasets
* Mobile app deployment using Streamlit Sharing or Docker
* Adding explainability using SHAP or Grad-CAM for clinical trust

### Results & Performance
#### Logistic Regression

[<img src="ANN_C.png">]()
