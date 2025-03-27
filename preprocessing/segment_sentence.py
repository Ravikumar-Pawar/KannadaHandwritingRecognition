import os
import cv2
import numpy as np

def segment_sentence(image, directory):

    line_dir = os.path.join(directory, "lines")
    os.makedirs(line_dir, exist_ok=True)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Dilate the image
    kernel = np.ones((5, 100), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours (Fixed: OpenCV 4+ returns only 2 values)
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[1])

    sentences = []

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)

        # Ignore small unwanted elements
        if w * h < 5000:
            continue

        # Extract region of interest (ROI)
        roi = thresh2[y:y+h, x:x+w]
        sentences.append(roi)

        # Save segmented image
        cv2.imwrite(os.path.join(line_dir, f"{i:02d}.png"), roi)

    return sentences
