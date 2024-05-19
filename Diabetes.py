import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import xgboost as xgb
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import streamlit_authenticator as stauth  # pip install streamlit-authenticator
import firebase_admin
from firebase_admin import credentials, auth, db
from firebase_admin import firestore
from st_pages import Page, show_pages, add_page_title
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Set the Streamlit configuration option to suppress the warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Function to load patient data and diabetes diagnosis data from various sources
def load_data():
    # For demonstration, let's just load a sample dataset
    file_path = st.file_uploader("Upload CSV file", type=["csv"])
    if file_path is not None:
        data = pd.read_csv(file_path)
        return data
    else:
        return None

# Function to clean and preprocess data (remove outliers, handle missing values)
def clean_and_preprocess_data(data):
    # For demonstration, let's just remove rows with missing values
    cleaned_data = data.dropna()
    return cleaned_data

# Function to encode categorical variables
def encode_categorical_variables(data):
    # No categorical variables to encode in this sample dataset
    return data

# Function to split data into training and testing sets
def split_data(data):
    # Split data into features (X) and target variable (y)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Function to train KNN model with hyperparameter tuning
def train_knn_with_tuning(X_train, X_test, y_train, y_test):
    param_grid = {'n_neighbors': range(1, 21)}  # Trying different values of n_neighbors from 1 to 20
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, best_knn

# Main function for data acquisition, preprocessing, and model training with tuning
def data_acquisition_preprocessing_and_training():
    st.title("Diabetes Prediction using KNN Classifier")
    st.write("## Data Acquisition and Preprocessing")

    # Load data
    data = load_data()
    if data is not None:
        st.write("### Dataset Preview:")
        st.write(data.head())

        # Clean and preprocess data
        cleaned_data = clean_and_preprocess_data(data)

        # Encode categorical variables
        encoded_data = encode_categorical_variables(cleaned_data)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = split_data(encoded_data)

        # Train KNN model with hyperparameter tuning
        accuracy, best_knn = train_knn_with_tuning(X_train, X_test, y_train, y_test)

        # Display results
        st.write('### Data after Pre-processing:')
        st.write(encoded_data.head())

        st.write('### Best KNN Model Accuracy after Hyperparameter Tuning:')
        st.write(f'{accuracy:.2%}')
        st.write('### Best Parameters:')
        st.write(best_knn.get_params())
    else:
        st.write("Please upload a CSV file to proceed.")
    
# Function for Exploratory Data Analysis (EDA)
def exploratory_data_analysis(data):
    st.title("Exploratory Data Analysis (EDA)")

    # Visualize distribution of diabetes risk levels
    with st.container():
        st.write("## Distribution of Diabetes Risk Levels")
        sns.set_theme()
        plt.figure(figsize=(6, 4))
        sns.countplot(x='Outcome', data=data)
        st.pyplot()

    # Analyze correlations between features and diabetes risk
    with st.container():
        st.write("## Correlation Heatmap")
        plt.figure(figsize=(8, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        st.pyplot()

    # Identify patterns and trends in patient data
    with st.container():
        st.write("## Pairplot of Features")
        plt.figure(figsize=(10, 8))
        sns.pairplot(data, hue='Outcome', diag_kind='kde')
        st.pyplot()
        
# Function to train KNN model and evaluate its performance
def train_and_evaluate_knn(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1
        
# Function for hyperparameter tuning
def hyperparameter_tuning(X_train, y_train):
    # Define parameter grid
    param_grid = {'n_neighbors': [3, 5, 7, 9]}

    # Initialize KNN classifier
    knn = KNeighborsClassifier()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5)

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Get best parameters
    best_params = grid_search.best_params_

    return best_params

# Main function for Model Building and Evaluation
def model_building_and_evaluation():
    st.title("Model Building and Evaluation")

    # Load data
    data = load_data()

    # Clean and preprocess data
    cleaned_data = clean_and_preprocess_data(data)

    # Encode categorical variables
    encoded_data = encode_categorical_variables(cleaned_data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = split_data(encoded_data)

    # Train and evaluate KNN model
    accuracy, precision, recall, f1 = train_and_evaluate_knn(X_train, X_test, y_train, y_test)

    # Display model evaluation metrics
    st.write("### Model Evaluation Metrics")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-score: {f1:.2f}")

    # Perform hyperparameter tuning
    best_params = hyperparameter_tuning(X_train, y_train)
    st.write("### Best Hyperparameters")
    st.write(best_params)

    
# initialize the Firebase app for Diabetes Prediction model
def initialize_diabetes_app():
    cred = credentials.Certificate('D:\\#YPR\\3rd Trimester\\Machine Learning\\Porjects\\Diabetes prediction\\ai-fusion-framework-firebase-adminsdk-n3qh4-ff9d6998b2.json')
    app = None
    try:
        app = firebase_admin.get_app('diabetes_app')
    except ValueError:
        pass
    if app is None:
        app = firebase_admin.initialize_app(cred, name='diabetes_app')
    diabetes_db = firestore.client(app)
    return diabetes_db 
diabetes_db = initialize_diabetes_app()

def run_ml_app():
    
    st.set_page_config(
        page_title="Modular Coding Predictive Suite",
        page_icon="	:robot_face:",
        layout="wide",
    )

    diabetes_model = pickle.load(open('D:/#YPR/3rd Trimester/Machine Learning/Porjects/Diabetes prediction/diabetes_model.sav', 'rb'))

    with st.sidebar:
        selected = option_menu('Predictive Suite',
                            ['Data Acquisition and Pre-processing', 
                             'Exploratory Data Analysis', 
                             'Model Building and Evaluation', 
                             'Diabetes Prediction'],
                            icons=[ 'database-fill','bar-chart' ,'cloud-upload', 'hospital'],
                            default_index=0)
         # Navigation and main content
    if selected == 'Home':
        st.title('Welcome to the ML Prediction System')
        st.write('This system provides predictions for various diseases and financial metrics.')
        st.write('Please select an option from the sidebar to get started.')
    
       
# Diabetes Prediction Page
    if (selected == 'Diabetes Prediction'):
        
        # page title
        st.title('Diabetes Prediction using ML')
        
        
        # getting the input data from the user
        col1, col2, col3 = st.columns(3)
        
        with col1:
            Pregnancies = st.text_input('Number of Pregnancies')
            
            
        with col2:
            Glucose = st.text_input('Glucose Level(80-200mg/dL)')
        
        with col3:
            BloodPressure = st.text_input('Blood Pressure value()')
        
        with col1:
            SkinThickness = st.text_input('Skin Thickness value')
        
        with col2:
            Insulin = st.text_input('Insulin Level')
        
        with col3:
            BMI = st.text_input('BMI value')
        
        with col1:
            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
        
        with col2:
            Age = st.text_input('Age of the Person')
        
        
        # code for Prediction
        diab_diagnosis = ''
        
        # creating a button for Prediction
        
        if st.button('Diabetes Test Result'):
            diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            
            # create a dictionary to store the input data
            input_data = {
                'pregnancies': Pregnancies,
                'glucose': Glucose,
                'blood_pressure': BloodPressure,
                'skin_thickness': SkinThickness,
                'insulin': Insulin,
                'bmi': BMI,
                'dpf': DiabetesPedigreeFunction,
                'age': Age,
                'diagnosis': diab_prediction[0]
            }
            input_data = {k: int(v) if isinstance(v, np.int64) else v for k, v in input_data.items()}
            # 'pregnancies': This attribute represents the number of times a patient has been pregnant. It is a known risk factor for developing diabetes in women.

            # 'glucose': This attribute represents the patient's fasting plasma glucose levels. High levels of glucose in the blood are a common symptom of diabetes.

            # 'blood_pressure': This attribute represents the patient's diastolic blood pressure. High blood pressure is a risk factor for developing diabetes.

            # 'skin_thickness': This attribute represents the thickness of the patient's skinfold at the triceps. Higher values may be associated with insulin resistance and diabetes.

            # 'insulin': This attribute represents the patient's insulin levels. Insulin is responsible for regulating glucose levels in the blood, and low levels may indicate diabetes.

            # 'bmi': This attribute represents the patient's body mass index, which is a measure of body fat based on height and weight. Obesity is a major risk factor for developing diabetes.

            # 'dpf': This attribute represents the patient's diabetes pedigree function, which is a measure of the patient's genetic risk for developing diabetes.

            # 'age': This attribute represents the patient's age. As people age, their risk of developing diabetes increases.

            # 'diagnosis': This attribute represents the predicted diagnosis of the patient (either "diabetic" or "non-diabetic") based on the other attributes in the dataset and the trained model.
            
            # store the input data in the database
            diabetes_db.collection('diabetes_inputs').add(input_data)
            if (diab_prediction[0] == 1):
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
            
        st.success(diab_diagnosis)
    elif (selected == 'Data Acquisition and Pre-processing'):
        data_acquisition_preprocessing_and_training()
        
    elif selected == 'Exploratory Data Analysis':
        st.title("Exploratory Data Analysis")

        # Load data
        data = load_data()

        # Perform EDA
        exploratory_data_analysis(data)
        
    elif selected == 'Model Building and Evaluation':
        model_building_and_evaluation()

    elif selected == 'Model Deployment and Monitoring':
        model_deployment_and_monitoring()

if __name__ == "__main__":
    run_ml_app()