import streamlit as st
import pandas as pd
import os

# pandas profiliing
from ydata_profiling import ProfileReport
from streamlit_ydata_profiling import st_profile_report

# automated ML pipeline using pycaret
from pycaret.regression import setup as setupr, compare_models as compare_models_r, pull as pull_r, save_model as save_model_r
from pycaret.classification import setup as setupc, compare_models as compare_models_c, pull as pull_c, save_model as save_model_c

st.title("Auto ML")
st.markdown("<br />", unsafe_allow_html=True)

with st.sidebar:
    st.title("Auto ML")
    choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
    st.info("This app allows you to automate the ML pipeline using streamlit and Pandas Profiling")

file_path="static/sourcedata.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path, index_col=None)

if choice == "Upload":
    st.subheader("Upload your dataset here")
    file = st.file_uploader("Upload your dataset")
    if file:
        st.write("Uploaded", file.name)
        skip_first_row = st.checkbox("Skip first row")
        skiprows = 1 if skip_first_row else 0
        df = pd.read_csv(file, skiprows=skiprows)
        df.to_csv("static/sourcedata.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.subheader("Automated Exploratory Data Analysis")
    if 'df' in locals() or 'df' in globals():
        profile_report = ProfileReport(df, title="Pandas Profiling Report")
        st_profile_report(profile_report, navbar=True)
    else:
        st.warning("Please upload a dataset to start the profiling")

if choice == "ML":
    st.subheader("AutoML Magic is here 🔥")
    if 'df' in locals() or 'df' in globals():
        problem_type = st.selectbox("Select Problem Type", ["Regression", "Classification"])
        target = st.selectbox("Select Your Target", df.columns)
        if st.button("Train Model"):
            if problem_type == "Classification":
                setupc(df, target=target)
                best_model = compare_models_c()
                setup_df = pull_c()
                st.info("This are the ML Experiment settings")
                st.dataframe(setup_df)
                compare_df = pull_c()
                st.info("This is the ML Model")
                st.dataframe(compare_df)
                st.write(f"The best model for the dataset is: {best_model.__class__.__name__}")
                best_model
                save_model_c(best_model, "models/best_model")
            else: # regression
                setupr(df, target=target)
                best_model = compare_models_r()
                setup_df = pull_r()
                st.info("This are the ML Experiment settings")
                st.dataframe(setup_df)
                compare_df = pull_r()
                st.info("This is the ML Model")
                st.dataframe(compare_df)
                st.write(f"The best model for the dataset is: {best_model.__class__.__name__}")
                best_model
                save_model_r(best_model, "models/best_model")
    else:
        st.warning("Please upload a dataset to start the Model Building")

if choice == "Download":
    st.subheader("Download the trained model here")
    if os.path.exists("models/best_model.pkl"):
        with open("models/best_model.pkl", 'rb') as f:
            st.download_button(f"Download the Model", f, "trained_model.pkl")
    else:
        st.warning("No trained model, train a model and then download it")

footer="""
<style>
    a:link , a:visited{
        color: red;
        background-color: transparent;
        text-decoration: underline;
    }

    a:hover,  a:active {
        color: white;
        background-color: transparent;
        text-decoration: underline;
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #262631;
        color: white;
        text-align: center;
    }
</style>

<div class="footer">
    <p>Developed with 🔥 by <a style='text-align: center; padding-left: 2px; padding-right: '2px';' href="https://linktr.ee/ashishnex007" target="_blank">Ashish Nex</a></p>
</div>
"""

st.markdown(footer,unsafe_allow_html=True)