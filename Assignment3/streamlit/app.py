import numpy as np
from numpy import array
import pandas as pd
import joblib
import streamlit as st 
import random as ran
from pathlib import Path
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from PIL import Image



# interact with FastAPI endpoint
backend = "http://api:8005/predict_model"


pickle_in = open("modelLinearRegression.pkl","rb")
reg_model = joblib.load(pickle_in)



with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("UI interface for Model as a service")
    choice = st.radio("Navigation", ["Home Page","Manually Checking Predictions","Profiling Any Dataset", "Model Performance"])
    st.info("This project application helps you build and explore your data.")


def predict_flashes(pixels):
    inputs = pixels.split(',')
    y_values= array([inputs]).reshape(-1,1)
    prediction = reg_model.predict(y_values)
    print(prediction)
    return list(prediction)

if choice == "Home Page":
    st.title("Basic training of a simple ML Regression model using a single feature/predictor/input")
    st.title("This model will predict how many lightning flashes are present in a particular image")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Model as a Service ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")


if choice == "Manually Checking Predictions": 
    st.info("You can now check flashes predictions by giving image pixels")
    flash_input = st.text_input("Enter 9 pixel", key="{}")
    result=""
    if st.button("Predict"):
        result= list(predict_flashes(flash_input))
    
    st.success('The output is {}'.format(result))


if choice == "Profiling Any Dataset": 
    st.title("Exploratory Data Analysis")
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
      #  df.to_csv('dataset.csv', index=None)
        st.dataframe(df)
    profile_df = df.profile_report()
    st_profile_report(profile_df)


if choice == "Model Performance":  
    if st.button("See Model Performance"):
        image = Image.open('slr1.png')
        st.image(image, caption='Simple ML Regression')
        st.info("You'll notice right here, there are ALOT of no flash images. You will see if we plot the number of flashes as a function of the minimum brightness temperature it might be very difficult to fit a linear method (i.e., Linear regression) to the data")






