import streamlit as st
import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model and the label binarizer
@st.cache(allow_output_mutation=True)
def load_model_and_labels():
    model = load_model('cnn_model.h5')
    label_binarizer = pickle.load(open('label_transform.pkl', 'rb'))
    return model, label_binarizer

model, label_binarizer = load_model_and_labels()

# Function to convert images to array
def convert_image_to_array(image_data):
    try:
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.resize(image, (256, 256))   
            return img_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# Streamlit app
st.title("Image Classification App")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an array
    image_data = uploaded_file.read()
    image_array = convert_image_to_array(image_data)

    if image_array.size == 0:
        st.error("Invalid image")
    else:
        # Normalize the image
        image_array = np.array(image_array, dtype=np.float16) / 255.0

        # Ensure the image_array has the correct shape (1, 256, 256, 3)
        image_array = np.expand_dims(image_array, axis=0)

        # Make a prediction
        prediction = model.predict(image_array)
        predicted_class = label_binarizer.inverse_transform(prediction)[0]

        st.success(f"Prediction: {predicted_class}")

        # Display the image
        st.image(image_data, caption='Uploaded Image', use_column_width=True)
