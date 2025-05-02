"""
Used to check camera functionality
"""

### Author: Edric Lay
### Date Created: 22/04/2025
############################################################################

import cv2 as cv
from datetime import datetime

def createVideoWriter(function: str, cameraProperties: list):
    """Creates a videoWriter object to be used to record a video

    Args:
        function (str): Name of the function of the video
        cameraProperties (list): Properties of the camera in the form [width,height,fps,fourcc]

    Returns:
        videoWriter (cv.VideoWriter): videoWriter object used to write to the video
        videoName (str): Name of the video being recorded to
    """
    [width,height,fps,fourcc] = cameraProperties
    date = datetime.now()
    dateStr = date.strftime("%d%m%y %H%M%S")
    videoName = f"video/{function}-{dateStr}.avi"
    videoWriter = cv.VideoWriter(videoName, fourcc, fps, (int(width),int(height)))
    return videoWriter, videoName

def getCameraProperties(cap: cv.VideoCapture):
    """Returns the properties of the camera after upgrading it if possible

    Args:
        cap (cv.VideoCapture): Camera being used to capture videos

    Returns:
        cameraProperties (list): Properties of camera being used in format = [width,height,fps,fourcc]
    """
    # Get camera properties
    width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv.CAP_PROP_FPS)
    fourcc = cv.VideoWriter.fourcc(*"XVID")  # Ensure this is an integer

    # try:
    #     # Try setting camera to be HD
    #     cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    #     cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
    #     cap.set(cv.CAP_PROP_FPS, 30)  # Try setting 30 FPS

    #     # Get back the actual settings
    #     width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    #     height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    #     fps = cap.get(cv.CAP_PROP_FPS)
    
    # except:
    #     pass

    cameraProperties = [width,height,fps,fourcc]
    print(cameraProperties)
    
    return cameraProperties

def recordVideo(videoWriter: cv.VideoWriter, cap: cv.VideoCapture, controller = None):
    """Function used to record a video for a videoWriter object

    Args:
        videoWriter (cv.VideoWriter): VideoWriter object for video to be written to
        cap (cv.VideoCapture): Camera being used to capture video
        controller (optional): Controller used to detect if video should stop recording. If no controller provided, it will record 600 frames.
    """
    print("Start recording video")
    if controller == None:
        index = 0
        while index < 600:
            ret, frame = cap.read()
            if ret:
                videoWriter.write(frame)
                cv.imshow("Recording", frame)

            if cv.waitKey(1) == ord('q'):
                break

            index +=1
    else:
        while controller.record == True:
            ret, frame = cap.read()
            if ret:
                videoWriter.write(frame)
                cv.imshow("Recording", frame)

            if cv.waitKey(1) == ord('q'):
                break
    
    print("Finished recording video")
    videoWriter.release()

def displayVideo(videoName: str):
    """Display the video on a local frame

    Args:
        videoName (str): Name of video being displayed
    """
    footage = cv.VideoCapture(videoName)
    print("Now playing video!!!")
    while True:
        try:
            ret, frame = footage.read()
            if ret:
                cv.imshow("Recording", frame)

            if not ret or cv.waitKey(1) == ord('q'):
                break
        except:
            break

    footage.release()

    
if __name__ == "__main__":
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened:
        print("Camera brokey")
        exit()

    cameraProperties = getCameraProperties(cap)

    # testWriter, testName = createVideoWriter("testVideo",cameraProperties)

    # recordVideo(testWriter, cap)
    # cap.release()

    # displayVideo(testName)

    while True:
        ret, frame = cap.read()
        if ret:
            cv.imshow("frame", frame)

            if cv.waitKey(1) == ord("q"):
                break