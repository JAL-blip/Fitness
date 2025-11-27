import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

st.write(''' Predicción de si estás en forma ''')
st.image("fit.jpg", caption="Forma.")

st.header('Datos de evaluación')

def user_input_features():
    age = st.number_input('Edad:', min_value=0, max_value=100, value=20, step=1)
    height_cm = st.number_input('Altura (cm):', min_value=100, max_value=250, value=170, step=1)
    weight_kg = st.number_input('Peso (kg):', min_value=20, max_value=250, value=70, step=1)
    heart_rate = st.number_input('Frecuencia cardiaca:', min_value=40, max_value=250, value=70, step=1)
    activity_index = st.number_input('Actividad:', min_value=0, max_value=400, value=30, step=1)
    blood_pressure = st.number_input('Presión de sangre:', min_value=0, max_value=200, value=30, step=1)
    sleep_hours = st.number_input('Horas dormidas:', min_value=0, max_value=30, value=8, step=1)
    nutrition_quality = st.number_input('Nutrición:', min_value=0, max_value=100, value=50, step=1)
    smokes = st.number_input('Fumar (0 = No, 1 = Si):', min_value=0, max_value=1, value=0, step=1)
    gender = st.number_input('Género (0 = Mujer, 1 = Hombre):', min_value=0, max_value=1, value=0, step=1)

    user_input_data = {'Edad': age,
                       'Altura': height_cm,
                       'Peso': weight_kg,
                       'Frecuencia_Cardiaca': heart_rate,
                       'Actividad': activity_index,
                       'Presion_de_sangre': blood_pressure,
                       'Horas_dormidas': sleep_hours,
                       'Nutrición': nutrition_quality,
                       'Fumar': smokes,
                       'Genero': gender}

    features = pd.DataFrame(user_input_data, index=[0])
    return features

df = user_input_features()


titanic = pd.read_csv('Fitness2.csv', encoding='latin-1')
X = titanic.drop(columns='is_fit')
Y = titanic['is_fit']


classifier = DecisionTreeClassifier(max_depth=6,criterion='gini',min_samples_leaf=50,max_features=5,random_state=1614372)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction[0] == 0:
    st.write('No está en forma')
elif prediction[0] == 1:
    st.write('Sí está en forma')
else:
    st.write('Sin predicción')
