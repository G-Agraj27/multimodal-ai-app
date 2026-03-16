import streamlit as st
import numpy as np
import cv2
import joblib

st.title("Multimodal AI Disease Detection System")

# Load models
#cnn_model = tf.keras.models.load_model("cnn_xray_model.h5")
patient_model = joblib.load("patient_model.pkl")

st.header("Upload Chest X-ray Image")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 0)

    st.image(img, caption="Uploaded X-ray Image", width=300)

    img = cv2.resize(img,(150,150))
    img = img/255.0
    img = img.reshape(1,150,150,1)

    image_pred = cnn_model.predict(img)

st.header("Enter Patient Clinical Data")

age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", [0,1])
cp = st.selectbox("Chest Pain Type (cp)", [0,1,2,3])
trestbps = st.number_input("Resting Blood Pressure")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("Fasting Blood Sugar >120 (1=True,0=False)", [0,1])
restecg = st.selectbox("Rest ECG", [0,1,2])
thalach = st.number_input("Max Heart Rate Achieved")
exang = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak")
slope = st.selectbox("Slope", [0,1,2])
ca = st.selectbox("Number of Major Vessels (ca)", [0,1,2,3])
thal = st.selectbox("Thal", [0,1,2,3])

patient_data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])

if st.button("Predict Disease"):

    patient_pred = patient_model.predict(patient_data)

    if image_pred > 0.5 or patient_pred == 1:
        st.error("Disease Detected")
    else:
        st.success("Normal")
