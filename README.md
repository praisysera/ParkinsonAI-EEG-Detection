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

| Model                        | Role                             |  Accuracy                  |     
|---------------------         |---------------------------------               |----------------------------|
| Logistic Regression	         | Baseline machine learning model	              | Moderate                   |
| PCA + ML Classifier          |Dimensionality reduction + classification	      | Good
| Artificial Neural Network	   | Deep learning model capturing complex patterns |	High
| Convolutional Neural Network |	Best performing DL model for spatial features	| Best (100%)
