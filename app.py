import pandas as pd
import numpy as np
import pickle
import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pickle', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
categories_geo = list(onehot_encoder_geo.categories_[0])

with open('label_encoder_gender.pickle', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pickle', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit UI
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', categories_geo)
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 100, 30)
balance = st.number_input('Balance', min_value=0.0)
credit_score = st.number_input('Credit Score', min_value=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0)
tenure = st.slider('Tenure', 0, 10, 5)
num_of_products = st.slider('Number of Products', 1, 4, 1)
has_credit_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input
input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_credit_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

input_df = pd.DataFrame([input_data])

# OneHotEncode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine all
input_full = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

# Ensure correct column order
input_full = input_full[scaler.feature_names_in_]

# Scale input
input_scaled = scaler.transform(input_full)

# Predict
prediction = model.predict(input_scaled)
churn = prediction[0][0] > 0.5

# Display
if churn:
    st.error('⚠️ Customer is likely to churn.')
else:
    st.success('✅ Customer is not likely to churn.')

st.write(f'**Churn probability:** {prediction[0][0]:.2f}')
