import streamlit as st
import pickle
import numpy as np
#IMPORT MODULES
#IMPORT MODULES
import pandas as pd

# SENTIMENT ANALYSIS USING VADER

from model_final import prepare_dataframe

def load_model():
    with open('Model/model_3.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data["MODEL"]
label_encoders = data["LABEL_ENCODERS"]
scaler = data["SCALER"]


def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("""### We need some information to predict the salary""")

    age = (
        "18-24 years old",
        "25-34 years old",
        "35-44 years old",
        "45-54 years old",
        "55-64 years old",
        "65 years or older",
        "Prefer not to say"
    )

    dev_type = (
        'Senior Executive (C-Suite, VP, etc.)', 
        'Developer, back-end', 
        'Developer, front-end', 
        'Developer, full-stack', 
        'System administrator', 
        'Developer, QA or test', 
        'Designer', 
        'Data scientist or machine learning specialist', 
        'Data or business analyst', 
        'Security professional', 
        'Research & Development role', 
        'Developer, mobile', 
        'Database administrator', 
        'Developer, embedded applications or devices', 
        'Developer, desktop or enterprise applications', 
        'Engineer, data', 
        'Product manager', 
        'Academic researcher', 
        'Cloud infrastructure engineer', 
        'Other (please specify):', 
        'Developer Experience', 
        'Engineering manager', 
        'DevOps specialist', 
        'Engineer, site reliability', 
        'Project manager', 
        'Blockchain', 
        'nan', 
        'Developer, game or graphics', 
        'Developer Advocate', 
        'Hardware Engineer', 
        'Educator', 
        'Scientist', 
        'Marketing or sales professional', 
        'Student'
    )

    orgsize = (
        '2 to 9 employees', 
        '5,000 to 9,999 employees', 
        '100 to 499 employees', 
        '20 to 99 employees', 
        '1,000 to 4,999 employees', 
        '10 to 19 employees', 
        '10,000 or more employees', 
        '500 to 999 employees', 
        'Just me - I am a freelancer, sole proprietor, etc.', 
        'I don’t know',
        "NAN"
    )

    aiselect = ('Yes', "No, and I don't plan to", 'No, but I plan to soon')

    remoteworkselect = ('Remote', 'Hybrid (some remote, some in-person)', 'In-person',"Other")

    currency = ('USD\tUnited States dollar', 'INR\tIndian rupee')
    education_level=('Bachelor’s degree (B.A., B.S., B.Eng., etc.)', 'Some college/university study without earning a degree', 'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)', 'Primary/elementary school', 'Professional degree (JD, MD, Ph.D, Ed.D, etc.)', 'Associate degree (A.A., A.S., etc.)', 'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)', 'Something else')
    education_input = st.selectbox("Education Level", education_level)
    age_input = st.selectbox("Age", age)
    dev_type_input = st.selectbox("Developer Type", dev_type)
    orgsize_input = st.selectbox("Organisation Size", orgsize)
    aiselect_input = st.selectbox("Do you currently use AI tools in your development process?", aiselect)
    currency_input = st.selectbox("Which currency do you use day-to-day? If your answer is complicated, please pick the one you're most comfortable estimating in", currency)
    experience_input = st.slider("Years of Experience", 0, 50, 3)
    yearscode_input = st.slider("Years of Coding Experience", 0, 50, 3)
    yearscodepro_input = st.slider("Year of Pro Coding Experience", 0, 50, 3)
    remotework_input= st.selectbox("'Current Work Situation Description'",remoteworkselect)
    databases_input = st.text_input("Databases you have worked with (separated by ;)", "")
    languages_input = st.text_input("Programming Languages you have worked with (separated by ;)", "")
    learning_sources_input = st.text_input("Learning Sources you have used (separated by ;)", "")

    ok = st.button("Calculate Salary")
    if ok:
        map_ =  {
           'Age': [age_input],
           'AISelect': [aiselect_input],
           'OrgSize':[orgsize_input],
           'DevType':[dev_type_input],
           'YearsCode':[yearscode_input],
           'WorkExp': [experience_input],
           'YearsCodePro': [yearscodepro_input],
           "RemoteWork": [remotework_input],
           'Currency': [currency_input],
           "EdLevel": [education_input],
           "LanguageHaveWorkedWith": [databases_input],
           "DatabaseHaveWorkedWith": [languages_input],
           "LearnCode": [learning_sources_input]
        }
        df = pd.DataFrame(map_)
        salary = prepare_dataframe(df, model, label_encoders, scaler)
        st.subheader(f"The estimated annual salary is ${salary[0]:.2f}")

show_predict_page()