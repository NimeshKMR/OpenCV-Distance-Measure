import cv2
import numpy as np
import imutils

#centimeters
KNOWN_DISTANCE = 80 
KNOWN_WIDTH = 14.5    

def find_marker(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cnts:
    # Filter contours by size and aspect ratio
     filtered_cnts = [c for c in cnts if cv2.contourArea(c) > 500]  # Minimum size threshold
    if filtered_cnts:
        c = max(filtered_cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)

        # Ensure the detected rectangle meets aspect ratio criteria
        width, height = rect[1]
        if 0.5 < width / height < 2.0:  # Adjust ratio as per marker's shape
            return rect
    return cv2.minAreaRect(c)
    

def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

# Load the reference image to calculate the focal length
ref_image = cv2.imread("pick.jpg")
if ref_image is None:
    raise ValueError("Reference image 'pick.jpg' not found.")
marker = find_marker(ref_image)
if marker is None:
    raise ValueError("Could not find marker in the reference image.")
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

print(focalLength)

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera
if not cap.isOpened():
    raise RuntimeError("Could not start the video capture.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    marker = find_marker(frame)
    if marker is not None:
        cm = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])

        # Draw the bounding box and display the distance
        box = cv2.boxPoints(marker)
        box = np.int32(box)
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
        cv2.putText(frame, f"{cm:.2f} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("Live Feed", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


print(focalLength)

cap.release()
cv2.destroyAllWindows()