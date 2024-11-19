import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(1)  # Use '0' for default laptop camera, '1' for external camera
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = "C:\\Users\\Dell\\Desktop\\sign language\\Data\\Yes"

# Check if the folder exists, create it if not
if not os.path.exists(folder):
    os.makedirs(folder)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image. Exiting...")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure crop doesn't go out of bounds
        if y - offset >= 0 and y + h + offset <= img.shape[0] and x - offset >= 0 and x + w + offset <= img.shape[1]:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            # Show the cropped and resized images
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

        # Show the original image
        cv2.imshow('Image', img)

    # Check if the "s" key is pressed for saving the image
    key = cv2.waitKey(1)
    if key == ord("s"):
        # Increment the counter and save the image only when "s" is pressed
        counter += 1
        cv2.imwrite(f'{folder}/Image_{counter}_{int(time.time())}.jpg', imgWhite)
        print(f"Image {counter} saved.")

    # Press 'q' to quit the loop
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()