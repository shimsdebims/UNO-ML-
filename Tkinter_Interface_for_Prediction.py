import cv2
import numpy as np
import pickle
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# Function to load models
def load_models(model_path="models/"):
    with open(model_path + "color_model.pkl", "rb") as f:
        color_model, le_color = pickle.load(f)
    with open(model_path + "type_model.pkl", "rb") as f:
        type_model, le_type = pickle.load(f)
    return color_model, type_model, le_color, le_type

# Function to extract features from image
def extract_features(image):
    # Convert image to grayscale, detect edges, and resize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    resized_image = cv2.resize(edges, (100, 150)).flatten()
    
    # Color histogram
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return np.concatenate([resized_image, hist.flatten()])

# Function to predict card color and type
def predict_card(image):
    features = extract_features(image).reshape(1, -1)
    color_pred = color_model.predict(features)
    type_pred = type_model.predict(features)
    predicted_color = le_color.inverse_transform(color_pred)[0]
    predicted_type = le_type.inverse_transform(type_pred)[0]
    return predicted_color, predicted_type

# Function to select image from file
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        image = cv2.imread(file_path)
        show_prediction(image)

# Function to capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        show_prediction(frame)
    else:
        messagebox.showerror("Error", "Failed to capture image from webcam.")

# Function to display prediction results
def show_prediction(image):
    global panel
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image_rgb)
    img = img.resize((250, 350), Image.LANCZOS)  # Updated to use Image.LANCZOS
    imgtk = ImageTk.PhotoImage(img)

    panel.configure(image=imgtk)
    panel.image = imgtk

    color, card_type = predict_card(image)
    label_result.config(text=f"Predicted Color: {color}\nPredicted Type: {card_type}")

# Load the trained models
color_model, type_model, le_color, le_type = load_models()

# Initialize Tkinter application
app = tk.Tk()
app.title("Card Classifier")

# UI Elements
panel = tk.Label(app)
panel.pack()

btn_select = tk.Button(app, text="Select Image", command=select_image)
btn_select.pack(side="left", padx=10)

btn_capture = tk.Button(app, text="Capture Image", command=capture_image)
btn_capture.pack(side="right", padx=10)

label_result = tk.Label(app, text="Predicted Color and Type will appear here.")
label_result.pack(pady=10)

# Run the application
app.mainloop()
