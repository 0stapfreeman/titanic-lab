import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('titanic_model.pkl')

# Define the Streamlit app
st.title('Titanic Survival Prediction')
st.write('This app predicts the survival of passengers on the Titanic.')

# Input fields for user data
pclass = st.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1, 2, 3])
age = st.slider('Age', 0, 100, 25)
sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)
parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)
fare = st.number_input('Passenger Fare', min_value=0.0, value=0.0, step=0.1)
sex = st.selectbox('Sex', ['male', 'female'])
embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])

# Preprocess user input
def preprocess_input(pclass, age, sibsp, parch, fare, sex, embarked):
    sex_male = 1 if sex == 'male' else 0
    embarked_C = 1 if embarked == 'C' else 0
    embarked_Q = 1 if embarked == 'Q' else 0
    embarked_S = 1 if embarked == 'S' else 0

    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Sex_male': [sex_male],
        'Embarked_C': [embarked_C],
        'Embarked_Q': [embarked_Q],
        'Embarked_S': [embarked_S]
    })

    # Add missing columns and ensure correct order
    TRAINING_COLUMNS = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
    for col in TRAINING_COLUMNS:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[TRAINING_COLUMNS]
    return input_data

# Predict survival
if st.button('Predict'):
    input_data = preprocess_input(pclass, age, sibsp, parch, fare, sex, embarked)
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    if prediction[0] == 1:
        st.success(f'The passenger is likely to survive with a probability of {prediction_proba[0][1]:.2f}.')
    else:
        st.error(f'The passenger is unlikely to survive with a probability of {prediction_proba[0][0]:.2f}.')