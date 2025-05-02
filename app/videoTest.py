### Author: Edric Lay
### Date Created: 17/03/2025
############################################################################
import cv2 as cv
from objectDetectionYOLO import ObjectDetectorYOLO
from handLandmark import handLandmarker
from datetime import datetime
import time

if __name__ == "__main__":
    # Step 0: Initialise everything
    date = datetime.now()
    dateStr = date.strftime("%d%m%y %H%M%S")
    videoName = f"{dateStr}.avi"
    annotatedVideoName = f"annotated{dateStr}.avi"

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error with webcam!!!")
        exit()
    
    # Get camera properties
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter.fourcc(*"XVID")  # Ensure this is an integer

    # print(f"Resolution: {int(width)}x{int(height)}")
    # print(f"FPS: {fps}")

    try:
        # Try setting a different resolution
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv.CAP_PROP_FPS, 30)  # Try setting 60 FPS

        # Read back the actual settings
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv.CAP_PROP_FPS)

        # print(f"New Resolution: {int(width)}x{int(height)}")
        # print(f"New FPS: {fps}")
    
    except:
        # print("Can't change frame rate")
        pass

    recordVideo = cv.VideoWriter(videoName, fourcc, fps, (int(width),int(height)))
    
    objectDetector = ObjectDetectorYOLO(modelPath="runs/detect/train6/weights/last.pt")
    handDetector = handLandmarker(modelPath="hand_landmarker.task")

    # Step 1: Capture Video
    while True:
        ret, frame = cap.read()
        if ret:
            recordVideo.write(frame)
            cv.imshow("Recording", frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    recordVideo.release()
    cv.destroyAllWindows()

    # Step 2: Get each model to run on the video
    inputVideo = cv.VideoCapture(videoName)
    ret = True

    while ret:
        ret, frame = inputVideo.read()
        timeStampMSEC = inputVideo.get(cv.CAP_PROP_POS_MSEC)

        objectDetector.detect(frame, timeStampMSEC)
        handDetector.detect(frame, timeStampMSEC)
    
    framesNoObjects = objectDetector.checkEmpty()
    framesNoHands = handDetector.checkEmpty()

    inputVideo.release()

    # # Step 3: Perform data analytics
    # a. Determine when sanitisation has ended
    # Occurs when frameNoObjects is no longer continous
    endStartSanitisationTime = 0 # TODO figure out how to do this.

    # b. Determine when object collection has ended
    # Occurs when allObjectsNeeded are collected
    endObjectCollectionTime = 5

    # c. Determine when doing whatever they are doing has ended
    # Get from sensor data somehow
    endTaskTime = 10

    # d. Determine when they are cleaning up
    # Time when space is emptied
    emptiedSpaceTime = 15

    # e. Final Sanitisation
    # Don't need to figure out time stamp since that will be end of video.     


    # Step 4: Annotate and create a final video
    originalVideo = cv.VideoCapture(videoName)
    annotatedVideo = cv.VideoWriter(filename=annotatedVideoName, fourcc=fourcc, fps=fps, frameSize=(width,height))
    
    ret = True
    index = 0
    while ret:
        ret, frame = originalVideo.read()
        timeStampMSEC = originalVideo.get(cv.CAP_PROP_POS_MSEC)
        annotatedFrame = frame

        if timeStampMSEC < endStartSanitisationTime:
            annotatedFrame = handDetector.paintImage(frame,index)

        elif timeStampMSEC < endObjectCollectionTime:
            handDetector.resetMask()
            annotatedFrame = objectDetector.visualise(frame,index)

        elif timeStampMSEC < endTaskTime:
            pass
        
        elif timeStampMSEC < emptiedSpaceTime:
            pass

        else:
            annotatedFrame = handDetector.paintImage(frame,index)

        annotatedVideo.write(annotatedFrame)
        index += 1

    # # Step 5: PRAISE THE MIGHTY SUN
    originalVideo.release()
    annotatedVideo.release()

    finalVideo = cv.VideoCapture(annotatedVideoName)

    while True:
        ret, frame = cap.read()
        if ret:
            recordVideo.write(frame)
            cv.imshow("Recording", frame)

        if cv.waitKey(1) == ord('q'):
            break
    
    exit()