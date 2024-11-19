import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load the model and labels
model_path = r"C:\Users\Dell\Desktop\converted_keras\keras_model.h5"
labels_path = r"C:\Users\Dell\Desktop\converted_keras\labels.txt"

# Load the Keras model
model = load_model(model_path)

# Load the class labels
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Initialize video capture and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Constants for image processing
offset = 20
imgSize = 300

def preprocess_image(img_crop):
    img_white = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    aspectRatio = img_crop.shape[0] / img_crop.shape[1]
    if aspectRatio > 1:
        k = imgSize / img_crop.shape[0]
        w_cal = math.ceil(k * img_crop.shape[1])
        img_resize = cv2.resize(img_crop, (w_cal, imgSize))
        w_gap = math.ceil((imgSize - w_cal) / 2)
        img_white[:, w_gap:w_cal + w_gap] = img_resize
    else:
        k = imgSize / img_crop.shape[1]
        h_cal = math.ceil(k * img_crop.shape[0])
        img_resize = cv2.resize(img_crop, (imgSize, h_cal))
        h_gap = math.ceil((imgSize - h_cal) / 2)
        img_white[h_gap:h_cal + h_gap, :] = img_resize
    return img_white

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from camera.")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgWhite = preprocess_image(imgCrop)

        # Prepare the image for model prediction
        imgWhite = imgWhite / 255.0  # Normalize the image
        imgWhite = np.expand_dims(imgWhite, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = model.predict(imgWhite)
        index = np.argmax(prediction[0])
        label = labels[index]

        # Draw on the output image
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Show images
        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite[0])  # Remove batch dimension for display

    # Show the final image output
    cv2.imshow('Image', imgOutput)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()
