import streamlit as st
import compute
import numpy as np
import cv2
import base64

# Define a function to process the image (you can keep this function as it is)
def process_image(image_file):
    fake_image = compute.generate(image_file).convert('RGB')
    fake_image = np.array(fake_image, dtype=np.uint8)

    # Convert the NumPy array to a JPEG image in memory (as a bytes object)
    _, img_bytes = cv2.imencode('.jpg', fake_image)

    # Convert the bytes object to a base64-encoded string
    fake_image_bs64 = base64.b64encode(img_bytes).decode('utf-8')

    return fake_image

def main():
    st.set_page_config(
        page_title="Whitebox Cartoonization Application",
        page_icon="🖼️",
        layout="wide"
    )

    st.title("Whitebox Cartoonization Application")
    st.write("Whitebox Cartoonization is a generative app employing a trained Generative Adversarial Network (GAN) to achieve the goal of animating any image.")
    st.write("Below are a few examples of animated images generated via my cartoonization app")

    # Display example images
    st.write("### Examples")
    col1, col2, col3, col4 = st.columns(4)

    example_images = [
        "static/images/gigi_cartoon.png",
        "static/images/pizza_cartoon.png",
        "static/images/oxford_cartoon.png",
        "static/images/dubai_cartoon.png"
    ]

    for i, img_path in enumerate(example_images):
        if i % 4 == 0:
            col1.image(img_path, caption="", use_column_width=True)
            col1.header("Gigi Hadid")
        elif i % 4 == 1:
            col2.image(img_path, caption="", use_column_width=True)
            col2.header("Pizza")
        elif i % 4 == 2:
            col3.image(img_path, caption="", use_column_width=True)
            col3.header('Oxford')
        else:
            col4.image(img_path, caption="", use_column_width=True)
            col4.header('Dubai')
    
    # Upload an image for processing
    st.write("### Upload Your Image")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=False)
        if st.button("Process Image"):
            fake_image_bs64 = process_image(uploaded_image)

            
            st.image(fake_image_bs64, caption="Animated", use_column_width=False)

if __name__ == "__main__":
    main()
