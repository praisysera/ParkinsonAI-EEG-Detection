import streamlit as st
import pickle
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import pandas as pd
from tensorflow.keras.losses import MeanSquaredError # type: ignore
import os

def load_models():
    with open('./models/scaler.pkl','rb') as f:
        scaler = pickle.load(f)
    model = load_model('./models/model.h5', custom_objects={'mse': MeanSquaredError()})

    return model, scaler

def preprocess(csv_file, scaler):   
    in_df = pd.read_csv(csv_file)
    in_df = scaler.transform(in_df)

    return in_df

def predict(in_df, model):
    pred = model.predict(in_df)
    y_pred_nn = (pred > 0.8).astype(np.float32)

    return "Parkinson's" if y_pred_nn==1 else "Healthy"

def main():
    st.title("Parkinson's Prediction Using EEG and Neural Networks")
    st.subheader("Upload a CSV file containing the EEG data..")
    csv_file  = st.file_uploader("Choose CSV file", type="csv")
    if st.button("Perform Inference"):
        model, scaler = load_models()
        if csv_file is not None:
            file_path = './' + csv_file.name
            with open(file_path, 'wb') as f:
                f.write(csv_file.getbuffer())
            in_df = preprocess(file_path, scaler)
            output = predict(in_df, model)
            st.write("Result: ", output)
            os.remove(file_path)
        else:
            st.error("Please upload a file...")

if __name__=="__main__":
    main()