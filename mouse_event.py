import numpy as np
import cv2

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        print(f"{x},{y}")
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        strXY = str(x) + "," + str(y)
        cv2.putText(img, strXY, (x, y), font, .5, (255,255,0), 2)
        cv2.imshow("image", img)
    
img = np.zeros((512,512,3), np.uint8)
cv2.imshow("image",img)

cv2.setMouseCallback("image", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows