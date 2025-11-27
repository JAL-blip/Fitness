import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.write(''' Predición de si estas en forma ''')
st.image("fit.jpg", caption="Forma.")

st.header('Datos de evaluación')

def user_input_features():
    age = st.number_input('Edad:', min_value=0, max_value=100, value=20, step=1)
    height_cm = st.number_input('Altura (cm):', min_value=100, max_value=250, value=170, step=1)
    weight_kg = st.number_input('Peso (kg):', min_value=20, max_value=250, value=70, step=1)
    heart_rate = st.number_input('Frecuencia cardiaca:', min_value=40, max_value=250, value=70, step=1)
    activity_index = st.number_input('Actividad:', min_value=0, max_value=400, value=30, step=1)
    blood_pressure = st.number_input('Presion de sangre:', min_value=0, max_value=200, value=30, step=1)
    sleep_hours = st.number_input('Horas dormidas:', min_value=0, max_value=30, value=30, step=1)
    nutrition_quality = st.number_input('Nutrición:', min_value=0, max_value=100, value=30, step=1)
    smokes = st.number_input('Fumar:', min_value=0, max_value=50, value=30, step=1)
    gender = st.number_input('genero:', min_value=0, max_value=2, value=30, step=1)

    user_input_data = {
        'Edad': age,
        'Altura': height_cm,
        'Peso': weight_kg,
        'Frecuencia_Cardiaca': heart_rate,
        'Actividad': activity_index,
        'Presion_de_sangre':blood_pressure,
        'Horas_dormidas':sleep_hours,
        'Nutrición':nutrition_quality,
        'Fumar':smokes,
        'genero':gender

    }

    features = pd.DataFrame(user_input_data, index=[0])

    return features

df = user_input_features()

fitness = pd.read_csv('Fitness2.csv', encoding='latin-1')
X = fitness.drop(columns='is_fit')
Y = fitness['is_fit']


classifier = DecisionTreeClassifier(max_depth=3,criterion='gini',min_samples_leaf=20,max_features=None,random_state=1614372)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
    st.write('No esta en forma')
elif prediction == 1:
    st.write('Si esta en forma')
else:
    st.write('Sin predicción')
