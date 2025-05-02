from ultralytics import YOLO
import numpy as np
import cv2 as cv

def visualise(frame,result):
        annotatedImage = np.copy(frame)
        boxes = result[0].boxes
        cls = result[0].boxes.cls
        conf = result[0].boxes.conf
        for i, box in enumerate(boxes.xyxy):  # Iterate with index
            className = result[0].names[int(cls[i])]
            confidence = conf[i]
            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])
            cv.rectangle(annotatedImage, (xA, yA), (xB, yB), (0, 255, 0), 2)
            imageText = className + str(confidence)
            cv.putText(annotatedImage, imageText, (xA,yA),fontFace = cv.FONT_HERSHEY_COMPLEX, fontScale = 1.5, color = (250,225,100))

        return annotatedImage

if __name__ == "__main__":
    testModel = YOLO(model = "runs/detect/train6/weights/best.pt")
    video = cv.VideoCapture(0)
    if not video.isOpened():
        print("Error opening camera")
    
    else:
        while True:
            ret, frame = video.read()        
            if ret:
                result = testModel.predict(frame)
                frame = visualise(frame,result)
                cv.imshow('frame', frame)
            
            if cv.waitKey(1) == ord('q'):
                break