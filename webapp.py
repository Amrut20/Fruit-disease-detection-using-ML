import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Function to create the MobileNetV2 architecture and load weights
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("C:\\Users\\amrut\\mobilenetv2_classification_model1.keras")
    return model

# Function for model prediction with resized input images
@st.cache_resource
def model_prediction(_model, test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(224, 224))  # Resize input image
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0) / 255.0  # Normalize the image array
    predictions = _model.predict(input_arr)
    return predictions

# Dictionary mapping class names to fruit disease descriptions
disease_descriptions = {
    "Blotch_Apple": "Apple blotch fungus disease is a condition characterized by sooty blotches on the apple's surface.",
    "Normal_Apple": "No visible signs of disease. The apple appears to be healthy.",
    "Rot_Apple": "Symptoms usually appear on the side of the apple directly exposed to the sun as small, circular brown lesions that change to sunken, dark brown lesions as they enlarge.",
    "Scab_Apple": "Apple scab is one of the most common diseases on apple trees and other plants in the apple family. This disease pops up everywhere apple-family plants grow."
}
diseas_escriptions = {}

# CSS for sidebar background and footer
custom_css = """
<style>
    [data-testid="stSidebar"] {
        background-color: #c40000;
        background-size: cover;
    }
    
    .main {
        background-image: url('https://img.freepik.com/premium-photo/natural-green-leaves-tree-branch-white-background-with-copy-space-text_129447-388.jpg');
        background-size: cover;
        opacity: 0.9; /* Reduced transparency for the background */
        background-repeat: no-repeat;
        padding: 2rem; /* Add padding to avoid content sticking to the edges */
    }
    
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #9d0000;
        text-align: center;
        padding: 10px 0;
        color: white; /* Set footer text color to white */
    }
</style>
"""

# Footer content
footer_content = """
<div class="footer">
    <p>@Amrut kadam <br> MCA 4th Sem 2023-24<br>KUD, DHARWAD 580003 PAVATE NAGAR</p>
</div>
"""

# Main function to run Streamlit app
def main():
    st.markdown(custom_css + footer_content, unsafe_allow_html=True)

    st.sidebar.title("Fruit Disease Detection")
    
    st.header("FRUIT DISEASE DETECTION SYSTEM")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        st.image(test_image, width=128, use_column_width=True, caption="Uploaded Image")
        if st.button("Predict"):
            # Load the model
            mobilenetv2 = load_model()

            # Make prediction
            predictions_mobilenetv2 = model_prediction(mobilenetv2, test_image)

            # Get the class index with the highest probability
            result_index_mobilenetv2 = np.argmax(predictions_mobilenetv2)

            # Class names for the specified classes
            class_names = [
                "Blotch_Apple", "Normal_Apple", "Rot_Apple", "Scab_Apple"
            ]

            # Display prediction result with black color
            if result_index_mobilenetv2 < len(class_names):
                disease_name = class_names[result_index_mobilenetv2]
                st.markdown(f"<h4 style='color: white;'><strong>Model predicts the disease as: {disease_name}.</strong></h4>", unsafe_allow_html=True)
                st.markdown(f"<p style='color: white;'><strong>Disease Description:</strong> {disease_descriptions[disease_name]}</p>", unsafe_allow_html=True)
            else:
                st.error("Prediction index out of range. Please check your model and class names.")

            # Confidence score plotting
            fig, ax = plt.subplots()
            confidence_scores = predictions_mobilenetv2[0]
            ax.bar(class_names, confidence_scores, color=['blue', 'green', 'red', 'orange'])
            ax.set_ylabel('Confidence Score')
            ax.set_title('Prediction Confidence for Fruit Disease')
            st.pyplot(fig)

            st.markdown(footer_content, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
