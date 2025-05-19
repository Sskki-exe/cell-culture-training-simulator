"""
Used to take photos for model
"""
### Author: Edric Lay
### Date Created: 6/04/2025
############################################################################

import cv2 as cv
from datetime import datetime

if __name__ == "__main__":
    cap = cv.VideoCapture(1)

    if cap.isOpened:
        while True:
            ret, frame = cap.read()

            if ret:
               cv.imshow('frame',frame)
               key = cv.waitKey(1)

               if key == ord('q'):
                   break
               
               elif key == ord('y'):
                    labelTime = datetime.now().strftime("%d-%m_%H-%M-%S")
                    filename = f"data/{labelTime}.jpg"
                    cv.imwrite(filename, frame)
