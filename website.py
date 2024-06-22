import streamlit as st
import joblib
import pandas as pd

model = joblib.load('best_ensemble_model.pkl')
scalar = joblib.load('scaler.pkl')

st.title('FIFA Prediction')

st.write('This is a simple FIFA prediction model. Please enter the required details to get the prediction')

#Figure a way to get input from the user

def training_attributes():
    at1 = st.sidebar
    at2 = st.sidebar

    values = {'name of columns I am using': at1, 
              

    }

    characteristics = pd.DataFrame(values, index = [0])
    return characteristics

#Use the scaler and model to Display the output to the user



