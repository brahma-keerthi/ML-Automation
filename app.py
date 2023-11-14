import streamlit as st
import pandas as pd
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model

# Creating sidebar
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Automated Machine Learning 📊")

    # navigation bar
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This application is used to build ML model without code.")


# accessing saved dataset
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)

    
# Upload page
if choice == "Upload":
    st.title("Upload dataset for modelling!")
    # uploading file
    file = st.file_uploader("Upload the dataset", type="csv")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)  # saves the dataset
        st.dataframe(df)  # prints dataframes to screen


# Profiling page
if choice == "Profiling":
    st.title("Automated Exploratory Data Analysis 🗺️")
    profile_report = df.profile_report()
    st_profile_report(profile_report) # Displays report on screen

# ML page
if choice == "ML":
    st.title("Machine Learning 🤖")
    # Select the target for building model
    target = st.selectbox("Select Your Target", df.columns) 
    if st.button('Run Modelling'): 
        experiment = setup(df, target=target, verbose=False)
        setup_df = pull()
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.dataframe(compare_df)
        save_model(best_model, 'best_model')

# Download page
if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")
