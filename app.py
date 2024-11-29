# from flask import Flask, request, render_template
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os

# app = Flask(__name__,)

# # Load pretrained model (replace with your model path)
# model = load_model('model.h5') 

# # Classes for prediction
# soil_types = [ 'Black Soil', 'Sandy Soil', 'Loamy Soil', 'Yellow Soil', 'Cinder Soil']

# # Configure upload folder
# UPLOAD_FOLDER = 'uploads'
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Handle the prediction of soil type."""
#     soil_image = request.files.get('soil_image')
#     soil_video = request.files.get('soil_video')

#     # Check if both files are uploaded
#     if soil_image and soil_video:
#         return render_template('index.html')

#     # Handle soil image prediction
#     if soil_image:
#         # Save the uploaded image
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], soil_image.filename)
#         soil_image.save(file_path)

#         # Preprocess the image
#         img = image.load_img(file_path, target_size=(224, 224))  # Adjust target size to your model
#         img_array = image.img_to_array(img) / 255.0
#         img_array = np.expand_dims(img_array, axis=0)

#         # Predict the soil type
#         prediction = model.predict(img_array)
#         soil_type = soil_types[np.argmax(prediction)]

#         # # Cleanup uploaded file
#         os.remove(file_path)

#         # Redirect to the appropriate HTML template based on prediction
#         if soil_type == 'Black Soil':
#             return render_template('black.html')
#         elif soil_type == 'Sandy Soil':
#             return render_template('sandy.html')
#         elif soil_type == 'Loamy Soil':
#             return render_template('loamy.html')
#         elif soil_type == 'Yellow Soil':
#             return render_template('yellow.html')
#         else:  # Cinder Soil
#             return render_template('cinder.html')

#     # Handle soil video upload
#     if soil_video:
#         return render_template('index.html', error="Video processing is not implemented yet.")

#     # If no file is uploaded
#     return render_template('index.html', error="No file uploaded. Please try again.")

# if __name__ == '__main__':
#     app.run(debug=True)






import os
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load pretrained model
model = load_model('model.h5')

# Classes for prediction
soil_types = ['Black Soil', 'Sandy Soil', 'Loamy Soil', 'Yellow Soil', 'Cinder Soil']

# Utility function to predict soil type
def predict_soil_type(image_file):
    # Save uploaded image temporarily
    file_path = os.path.join("temp", image_file.name)
    with open(file_path, "wb") as f:
        f.write(image_file.read())

    # Preprocess the image
    img = image.load_img(file_path, target_size=(224, 224))  # Adjust target size to your model
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict soil type
    prediction = model.predict(img_array)
    soil_type = soil_types[np.argmax(prediction)]

    # Remove temporary file
    os.remove(file_path)

    return soil_type

# Main Streamlit app
def main():
    st.title("Soil Type Prediction")
    st.write(
        "Upload an image of soil, and the app will predict its type. "
        "Currently supported soil types: Black Soil, Sandy Soil, Loamy Soil, Yellow Soil, and Cinder Soil."
    )

    # File uploader for soil images
    uploaded_file = st.file_uploader("Choose a soil image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Soil Image", use_column_width=True)

        # Predict button
        if st.button("Predict"):
            soil_type = predict_soil_type(uploaded_file)
            st.subheader(f"Predicted Soil Type: {soil_type}")

            # Show additional information based on soil type
            if soil_type == "Black Soil":
                st.markdown(open("black.html").read(), unsafe_allow_html=True)
            elif soil_type == "Sandy Soil":
                st.markdown(open("sandy.html").read(), unsafe_allow_html=True)
            elif soil_type == "Loamy Soil":
                st.markdown(open("loamy.html").read(), unsafe_allow_html=True)
            elif soil_type == "Yellow Soil":
                st.markdown(open("yellow.html").read(), unsafe_allow_html=True)
            else:  # Cinder Soil
                st.markdown(open("cinder.html").read(), unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    if not os.path.exists("temp"):
        os.makedirs("temp")  # Temporary directory for uploaded files
    main()
