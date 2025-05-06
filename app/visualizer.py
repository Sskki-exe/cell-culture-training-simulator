import pandas as pd
import cv2 as cv
import numpy as np
import open3d as o3d
from camera import createVideoWriter, getCameraProperties

def visualizerVideo(filename: str, cameraProperties: list):
    data = pd.read_csv(filename)
    videoWriter, videoName = createVideoWriter("model", cameraProperties)

    buttonUp = o3d.io.read_triangle_mesh("visualiser/pipette_up.obj")
    buttonDown = o3d.io.read_triangle_mesh("visualiser/pipette_down.obj")

    vis = o3d.visualisation.Visualizer()
    vis.create_window(visible=False, width=cameraProperties[0], height=cameraProperties[1])

    prevTransform = np.eye(4)

    for rowIndex, data in data.iterrows():
        pass

    return videoName


if __name__=="__main__":
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened:
        print("Camera brokey")
        exit()

    cameraProperties = getCameraProperties(cap)

    cap.release()
