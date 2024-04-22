# Diabetes Prediction Web App

This web application predicts whether a person is diabetic based on their medical information. It utilizes a Support Vector Machine (SVM) model trained on a dataset of diabetes patients.

## Usage

To use the web app, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the web app using Streamlit by running `streamlit run app.py` in your terminal.
4. Enter the required medical information in the sidebar.
5. Click the "Predict" button to see the prediction.

## Data Source

The dataset used for training the SVM model is available on [Kaggle](https://www.kaggle.com/datasets/mathchi/diabetes-data-set). It contains various medical features such as glucose level, blood pressure, BMI, etc., along with the target variable indicating whether the person is diabetic or not.

## Model

The SVM model used for prediction is trained using scikit-learn library. The model achieves an accuracy of 72% on the test data.


