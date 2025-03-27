import cv2

def sort_contours(cnts, method="left-to-right"):
    """Sort contours based on the specified method (left-to-right, top-to-bottom, etc.)."""
    reverse = method in ["right-to-left", "bottom-to-top"]
    axis = 1 if method in ["top-to-bottom", "bottom-to-top"] else 0

    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    sorted_ctrs = sorted(zip(cnts, bounding_boxes), key=lambda b: b[1][axis], reverse=reverse)
    
    return zip(*sorted_ctrs)  # Unzips into (sorted_contours, sorted_bounding_boxes)

def segment_character(image, directory):
    """Segments individual characters and ottaksharas from the given word image."""
    
    row, col = image.shape

    # Apply binary thresholding
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    _, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours (Fixed for OpenCV 4+)
    ctrs, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours left-to-right for correct character order
    ctrs, bounding_boxes = sort_contours(ctrs, method="left-to-right")

    characters = {}
    ottaksharas = {}

    count = 0

    for i, (cnt, (x, y, w, h)) in enumerate(zip(ctrs, bounding_boxes)):
        # Ignore small contours (noise)
        if (w * h) < 100:
            continue

        # Extract character region
        roi = thresh2[y:y+h, x:x+w]

        """
        Ottakshara Handling:
        - Ottaksharas (subscript characters) are usually positioned lower than the main characters.
        - We assume that any contour starting **below 50%** of the image height is an ottakshara.
        - They are mapped to the previous main character (if any).
        """
        if y > (row / 2):  
            # If there's a previous character, associate the ottakshara with it
            if count > 0:
                ottaksharas[count - 1] = roi
        else:
            characters[count] = roi
            count += 1  # Increment only for main characters

    return characters, ottaksharas
