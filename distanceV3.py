import cv2
import numpy as np
import imutils

#centimeters
KNOWN_DISTANCE = 80  
KNOWN_WIDTH = 14.5      
KNOWN_HEIGHT = 10

def find_marker(image):
    global cnts
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) == 0:
        return None
    
    filtered_cnts = [c for c in cnts if cv2.contourArea(c) > 500]  
    if filtered_cnts:
        c = max(filtered_cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        width, height = rect[1]
        if 0.3 < width / height < 3.0:  
            return rect
    

def distance_to_camera(knownArea, focalLength, perArea):
    return (knownArea * focalLength) / perArea


ref_image = cv2.imread("pick.jpg")
if ref_image is None:
    raise ValueError("Reference image 'pick.jpg' not found.")
marker = find_marker(ref_image)
if marker is None:
    raise ValueError("Failed to find marker in the reference image.")
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH

cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    raise RuntimeError("Failed to start the video capture.")

while True:
    ret, frame = cap.read() 
    if not ret:
        print("Failed to capture frame.")
        break 

    marker = find_marker(frame)    
    if marker is not None:
        width, height = marker[1]
        perceived_area = width * height
        cm = distance_to_camera(KNOWN_WIDTH * KNOWN_HEIGHT, focalLength, perceived_area)

        box = cv2.boxPoints(marker)  
        box = np.int32(box) 
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)   
        cv2.putText(frame, f"{cm:.2f} cm", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("Live Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(focalLength)
cap.release()
cv2.destroyAllWindows()
