import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Load pre-trained classifiers from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
glasses_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Streamlit app
st.title("Object Detection App using opencv")

# Detection options
detection_type = st.selectbox("Select Detection Type", [
    "Face Detection", "Left Eye Detection", "Right Eye Detection", "Body Detection", 
    "Car Detection", "Car License Plate Detection", "Glasses Detection", "Smile Detection", "Bike Detection"])

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # Check if the image is in grayscale (1 channel) or color (3 channels)
    if len(img_array.shape) == 2:  # Grayscale image
        gray = img_array
    else:  # Color image
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Perform detection based on selected option
    if detection_type == "Face Detection":
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

    elif detection_type == "Left Eye Detection" or detection_type == "Right Eye Detection":
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            color_face_roi = img_array[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(face_roi)
            for (ex, ey, ew, eh) in eyes:
                if detection_type == "Left Eye Detection" and ex < w // 2:  # Detect left eye
                    cv2.rectangle(color_face_roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                elif detection_type == "Right Eye Detection" and ex >= w // 2:  # Detect right eye
                    cv2.rectangle(color_face_roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    elif detection_type == "Body Detection":
        bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in bodies:
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 0, 255), 2)

    elif detection_type == "Car Detection":
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in cars:
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 255, 0), 2)

    elif detection_type == "Car License Plate Detection":
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in plates:
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 255), 2)

    elif detection_type == "Glasses Detection":
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            color_face_roi = img_array[y:y + h, x:x + w]
            glasses = glasses_cascade.detectMultiScale(face_roi)
            for (gx, gy, gw, gh) in glasses:
                cv2.rectangle(color_face_roi, (gx, gy), (gx + gw, gy + gh), (0, 255, 255), 2)

    elif detection_type == "Smile Detection":
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            color_face_roi = img_array[y:y + h, x:x + w]
            smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(color_face_roi, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)

    elif detection_type == "Bike Detection":
        # Note: Bike detection is not included by default in OpenCV's Haar cascades.
        # You need a custom-trained model (like YOLO or SSD) to detect bikes. 
        st.warning("Bike detection not available with default Haar cascades. Consider using a custom model.")

    # Display the output
    st.image(img_array, caption=f"Detected objects - {detection_type}", use_column_width=True)
