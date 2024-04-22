import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained SVM model
model = joblib.load("model/svm_diabetes_model.pkl")

# Load the dataset
data = pd.read_csv("data/diabetes.csv")
X = data.drop(columns="Outcome", axis=1)

# Function to make predictions
def predict_diabetes(input_data):
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown(
    """
    <style>
    .header {
        background-color: #4CAF50;
        color: white;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        border-radius: 10px;
    }
    .prediction {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
    }
    .result {
        font-size: 20px;
        font-weight: bold;
        margin-top: 20px;
    }
    .resources {
        margin-top: 30px;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-around;
        align-items: center;
    }
    .resource-card {
        width: 300px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 0px 10px 0px rgba(0, 0, 0, 0.1);
        margin: 10px;
        padding: 20px;
    }
    .resource-card img {
        width: 200px;  /* Adjusted width */
        height: auto;  /* Maintain aspect ratio */
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .resource-card h3 {
        text-align: center;
        font-size: 18px;
        margin-bottom: 10px;
    }
    .resource-card p {
        text-align: justify;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown("<h1 class='header'>Diabetes Prediction</h1>", unsafe_allow_html=True)

# Sidebar with input fields
st.sidebar.markdown("<h2>Enter Patient Information</h2>", unsafe_allow_html=True)
pregnancies = st.sidebar.number_input("Number of Pregnancies", min_value=0, step=1)
glucose = st.sidebar.slider("Glucose Level", min_value=0, max_value=200, step=1)
blood_pressure = st.sidebar.slider("Blood Pressure", min_value=0, max_value=150, step=1)
skin_thickness = st.sidebar.slider("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.sidebar.slider("Insulin Level", min_value=0, max_value=300, step=1)
bmi = st.sidebar.slider("BMI", min_value=0.0, max_value=70.0, step=0.1)
diabetes_pedigree = st.sidebar.slider("Diabetes Pedigree Function", min_value=0.0, max_value=10.0, step=0.001)
age = st.sidebar.slider("Age", min_value=0, max_value=100, step=1)

# Make prediction when user clicks the button
if st.sidebar.button("Predict"):
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]).reshape(1, -1)
    prediction = predict_diabetes(input_data)
    if prediction == 0:
        st.markdown("<div class='prediction'><p class='result'>Based on the provided information, the person is predicted to be not diabetic.</p></div>", unsafe_allow_html=True)
    elif prediction == 1:
        st.markdown("<div class='prediction'><p class='result'>Based on the provided information, the person is predicted to be diabetic.</p></div>", unsafe_allow_html=True)
        st.markdown("<h2 class='resources'>Resources for Diabetes Management:</h2>", unsafe_allow_html=True)
        st.markdown("<div class='resources'>", unsafe_allow_html=True)
        
        # Resource 1
        st.image("images/diabetes_resource_1.jpg", use_column_width=True)
        st.markdown("<h3>American Diabetes Association</h3>", unsafe_allow_html=True)
        st.markdown("<p>The American Diabetes Association is a nonprofit organization that leads the fight against the deadly consequences of diabetes and fights for those affected by diabetes.</p>", unsafe_allow_html=True)
        st.markdown("<a href='https://www.diabetes.org/' target='_blank'>Learn More</a>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Resource 2
        st.image("images/diabetes_resource_2.jpg", use_column_width=True)
        st.markdown("<h3>National Institute of Diabetes and Digestive and Kidney Diseases</h3>", unsafe_allow_html=True)
        st.markdown("<p>The National Institute of Diabetes and Digestive and Kidney Diseases conducts and supports research on many of the most serious diseases affecting public health.</p>", unsafe_allow_html=True)
        st.markdown("<a href='https://www.niddk.nih.gov/' target='_blank'>Learn More</a>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Resource 3
        st.image("images/diabetes_resource_3.jpg", use_column_width=True)
        st.markdown("<h3>Centers for Disease Control and Prevention - Diabetes</h3>", unsafe_allow_html=True)
        st.markdown("<p>The Centers for Disease Control and Prevention (CDC) provides information and resources on diabetes prevention, management, and research.</p>", unsafe_allow_html=True)
        st.markdown("<a href='https://www.cdc.gov/diabetes/index.html' target='_blank'>Learn More</a>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
