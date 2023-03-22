import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Define the class labels
class_labels = ['Mild Demented', 'Moderate Demented',
                'Non Demented', 'Very Mild Demented']

# Define the function to preprocess the image


def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((176, 176))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Define the Streamlit app


def app():
    st.title('Alzheimer Detection App')
    st.write('Please enter your personal details along with MRI scan.')

    # Add fields for name, age, contact, and gender
    with st.form(key='myform', clear_on_submit=True):
        name = st.text_input('Name')
        age = st.number_input('Age', min_value=0, max_value=150)
        gender = st.radio('Gender', ('Male', 'Female'))
        contact = st.text_input('Contact')
        submit = st.form_submit_button("Submit")
        if submit:
            st.success(
                'Your personal information has been recorded. Please Upload the MRI Scan.', icon="✅")

    file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if file is not None:
        image = Image.open(file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        # Use the fields for name, age, contact, and gender in the output
        st.write('Name:', name)
        st.write('Age:', age)
        st.write('Gender:', gender)
        st.write('Contact:', contact)
        image = preprocess_image(image)
        prediction = model.predict(image)
        prediction = np.argmax(prediction, axis=1)
        st.success('Your prediction is: ' + class_labels[prediction[0]])


# Run the app
if __name__ == '__main__':
    app()
