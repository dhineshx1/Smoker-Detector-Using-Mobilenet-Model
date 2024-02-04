import tensorflow as tf 
from tensorflow import keras
import numpy as np
from PIL import Image,ImageOps
import streamlit as st


model = keras.models.load_model("model\model_one.h5")
def img_preprocess(img):
    size = (250,250)
    img = ImageOps.fit(img,size)
    img_array = np.array(img)
    img_array = img_array.reshape(1, 250, 250, 3)
    preprocess_input = keras.applications.mobilenet_v3.preprocess_input
    img_array = preprocess_input(img_array)
    return img_array

def predict_class(img):
    class_lable = ["Not Smoking","Smoking"]
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    return class_lable[predicted_class]



def app():
    st.set_page_config(page_title="SMOKER DETECTION API")
    st.header("SMOKER DETECTION")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image=""   
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.")
        submit=st.button("DETECT")
        if submit:
            image_data = img_preprocess(image)
            class_name =predict_class(image_data)
            st.subheader("Result....")
            if class_name =="Not Smoking":
                st.success(class_name)
            else:
                st.error(class_name)
app()