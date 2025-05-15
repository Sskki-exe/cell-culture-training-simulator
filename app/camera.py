"""
Used to check camera functionality
"""

### Author: Edric Lay
### Date Created: 22/04/2025
############################################################################

import cv2 as cv
import os
from datetime import datetime

K = [[812.95647633,   0.,         322.3390025],
 [  0.,         818.92425671, 311.69425812],
 [  0.,           0.,           1.,        ]]


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
    print(f"Video Properties: {cameraProperties}")
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
    print(cameraProperties)

    # testWriter, testName = createVideoWriter("testVideo",cameraProperties)

    # recordVideo(testWriter, cap)
    # cap.release()

    # displayVideo(testName)

    save_dir = "calibration_images"
    os.makedirs(save_dir, exist_ok=True)

    img_counter = 0

    print("Press SPACE to capture image, ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv.imshow("Camera", frame)

        key = cv.waitKey(1)

        if key % 256 == 27:  # ESC key
            print("Exiting...")
            break
        elif key % 256 == 32:  # SPACE key
            img_name = f"{save_dir}/image_{img_counter:03}.jpg"
            cv.imwrite(img_name, frame)
            print(f"{img_name} saved.")
            img_counter += 1

    # Release resources
    cap.release()
    cv.destroyAllWindows()

    # while True:
    #     ret, frame = cap.read()
    #     if ret:
    #         cv.imshow("frame", frame)

    #         if cv.waitKey(1) == ord("q"):
    #             break