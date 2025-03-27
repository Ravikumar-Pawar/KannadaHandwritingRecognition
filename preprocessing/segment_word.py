import os
import cv2
import numpy as np

def segment_word(image, directory, count):

    word_dir = os.path.join(directory, "words")
    os.makedirs(word_dir, exist_ok=True)

    # Apply thresholding
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Dilate the image
    kernel = np.ones((5, 40), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours (Fixed OpenCV issue)
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours from left to right
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

    words = []

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)

        # Ignore very small elements
        if (w * h) < 1000:
            continue

        # Extract word region
        roi = thresh2[y:y+h, x:x+w]

        # Save the segmented word
        words.append(roi)
        cv2.imwrite(os.path.join(word_dir, f"{count:02d}-{i:02d}.png"), roi)

    return words
