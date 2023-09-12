from flask import Flask, render_template, request
import numpy as np
import compute
import cv2, base64, os, io, warnings
from PIL import Image
import matplotlib.pyplot as plt
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@app.route("/", methods=["POST", "GET"])
def process():
    if request.method == 'POST':
        image_file = request.files.get('image')  # get the image files
        fake_image = compute.generate(image_file).convert('RGB')

        # Create a temporary directory to store the images
        temp_dir = os.path.join(BASE_DIR, 'static/temp_images')
        os.makedirs(temp_dir, exist_ok=True)
        
        #save_processed_image(fake_image, temp_dir, image_file.filename)
        # Convert the PIL image to cv2 format (NumPy array)
        fake_image = np.array(fake_image, dtype=np.uint8)
        
        # Convert the NumPy array to a JPEG image in memory (as a bytes object)
        _, img_bytes = cv2.imencode('.jpg', fake_image)

        # Convert the bytes object to a base64-encoded string
        fake_image_bs64 = base64.b64encode(img_bytes).decode('utf-8')

        context = {
            'fake_image': f"data:image/jpeg;base64,{fake_image_bs64}"
        }
        return render_template("index.html", **context)
    else:
        fake_image = None
        return render_template("index.html")

def save_processed_image(image_array, folder_path, filename):
    # Ensure the folder path exists, create it if it doesn't
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Convert the NumPy ndarray to a Pillow Image
    processed_image = Image.fromarray(np.uint8(image_array))
    
    # Save the image in the specified folder with the given filename
    save_path = os.path.join(folder_path, filename)
    processed_image.save(save_path)

@app.route("/about")
def about():
    name = "MOAZ MOHAMMED HUSAIN"
    return render_template("about.html", name=name)

if __name__ == "__main__":
    app.run(debug=True)