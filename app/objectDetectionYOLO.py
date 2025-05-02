"""
YOLO Code is adapted from:
https://docs.ultralytics.com/models/yolo11/#usage-examples
https://docs.ultralytics.com/modes/train/

"""
### Author: Edric Lay
### Date Created: 17/03/2025
############################################################################
### Library Imports
from ultralytics import YOLO
import numpy as np
import cv2 as cv
from result import Result

class ObjectDetectorYOLO():
    ############################
    # Values used for annotating on the image. Values taken from: 
    # https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    ############################
    def __init__(self, modelPath: str = "runs/detect/train6/weights/best.pt"):
        """Used to create Object Detector 

        Args:
            modelPath (str): Path for the model
        """
        self.model = YOLO(model = modelPath, verbose=False)
        self.model.to('cpu')
        self.classNames = self.model.names

    ################################ Video Analysis ############################################ 
    def analyseVideo(self, videoName: str):
        """Perform a object detection analysis of a full video

        Args:
            videoName (str): File name of the video

        Returns:
            videoAnalysis (list): List of results of each frame analyzed
        """
        footage = cv.VideoCapture(videoName)
        videoAnalysis = []

        while True:
            try:
                ret, frame = footage.read()
                if ret:
                    timeStamp = footage.get(cv.CAP_PROP_POS_MSEC)
                    frameResult = self.detect(frame, timeStamp)
                    videoAnalysis.append(frameResult)

                if not ret or cv.waitKey(1) == ord('q'):
                    break
            except:
                break

        footage.release()

        return videoAnalysis
    
    def detect(self, frame, timeStamp):
        """Method used to detect all objects in a frame

        Args:
            frame: Frame in which we will be running the detector on
            timeStamp: Timestamp of frame
        
        Returns:
            result (Result): Result of the frame analyzed
        """
        result = self.model.predict(frame, verbose = False)
        return Result(timeStamp, result[0])
    
    ############################### Drawing Tools #############################################
    def updateSpecificItemFrames(self, resultList: list, itemName: str):
        """Updates the resultList frames to show a specific item tracking based off a list of results

        Args:
            result (List(Result)): List of Result objects created after analysing a video 
            itemName (str): Specific name of item being looked for

        Returns:
            videoName (str): Name of the video with annotations
            count (int): Number of times that item has been removed
        """
        prevItemFound = False
        count = 0

        for analysis in resultList:
            annotated_image, status = self.visualiseSpecificItem(analysis, itemName)
            analysis.frame = annotated_image
            curItemFound = status
            if curItemFound != True and prevItemFound != True:
                count += 1
                analysis.status = False

            prevItemFound = curItemFound

        return count        

    def visualiseSpecificItem(self, frame, analysis: Result, itemName: str):
        """ Annotates frame when a specific item is detected

        Args:
            analysis (Result): Result object which is a frame which has been analysed
            itemName (str): Specific name of item being looked for

        Returns:
            annotated_image: Image with specific item labelled
            found (bool): Status if item is found
        """
        annotated_image = frame
        result = analysis.result
        boxes = result.boxes
        cls = result.boxes.cls
        found = False
        for i, box in enumerate(boxes.xyxy):  
            className = result[0].names[int(cls[i])]
            if className == itemName:
                found = True
                xA = int(box[0])
                yA = int(box[1])
                cv.putText(annotated_image, className, (xA,yA),fontFace = cv.FONT_HERSHEY_COMPLEX, fontScale = 1.5, color = (250,225,100))
        
        # cv.putText(annotated_image, "Performing Object Detection",
        # (0, 25), cv.FONT_HERSHEY_DUPLEX,
        # ObjectDetectorYOLO.FONT_SIZE, ObjectDetectorYOLO.HANDEDNESS_TEXT_COLOR, ObjectDetectorYOLO.FONT_THICKNESS, cv.LINE_AA)
        
        # cv.putText(annotated_image, f"Found {itemName}: {str(found)}",
        # (0, 50), cv.FONT_HERSHEY_DUPLEX,
        # ObjectDetectorYOLO.FONT_SIZE/2, ObjectDetectorYOLO.HANDEDNESS_TEXT_COLOR, ObjectDetectorYOLO.FONT_THICKNESS, cv.LINE_AA)

        return annotated_image, found

    def updateAllItemFrames(self, resultList: list):
        """Updates the resultList frames to show all items based off a list of results

        Args:
            result (List(Result)): List of Result objects created after analysing a video 

        Returns:
            videoName (str): Name of the video with annotations
        """

        for analysis in resultList:
            annotated_image, status = self.visualiseAll(analysis)
            analysis.frame = annotated_image

        return None        

    def visualiseAll(self, frame, analysis: Result):
        """ Annotates frame with all detected items

        Args:
            analysis (Result): Result object which is a frame which has been analysed

        Returns:
            annotated_image: Image with all known items labelled
        """
        annotated_image = np.copy(frame)
        result = analysis.result
        boxes = result.boxes
        cls = result.boxes.cls

        for i, box in enumerate(boxes.xyxy):  
            className = result[0].names[int(cls[i])]
            xB = int(box[2])
            xA = int(box[0])
            yB = int(box[3])
            yA = int(box[1])
            cv.rectangle(annotated_image, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv.putText(annotated_image, className, (xA,yA),fontFace = cv.FONT_HERSHEY_COMPLEX, fontScale = 1, color = (250,225,100))

        cv.putText(annotated_image, "Performing Object Detection",
        (0, 25), cv.FONT_HERSHEY_DUPLEX,
        ObjectDetectorYOLO.FONT_SIZE, (0,0,0), ObjectDetectorYOLO.FONT_THICKNESS, cv.LINE_AA)

        items = self.count(analysis)
        count = sum(items.values())

        cv.putText(annotated_image, f"Items found: {count}",
        (0, 50), cv.FONT_HERSHEY_DUPLEX,
        ObjectDetectorYOLO.FONT_SIZE/2, (0,0,0), ObjectDetectorYOLO.FONT_THICKNESS, cv.LINE_AA)

        return annotated_image, items, count

    #################################### Object Detection ########################################
    def count(self, analysis: Result):
        """ Get all classes detected and the amount of them

        Returns:
            countDict (dict): dictionary containing all present classes and the count of each
        """
        countDict = dict()
        detectedObjects = analysis.result.boxes.cls.numpy().astype(int) # get all classes as int

        for objectID in detectedObjects:
            className = self.classNames[objectID]
            if className in countDict:
                countDict[className] += 1
            else:
                countDict[className] = 1

        return countDict
    
    # def findItem(self,desiredObject: str):
    #     """Finds all instances of a desired object

    #     Args:
    #         desiredObject (str): Name of class we are looking for

    #     Returns:
    #         False, if desired object not in frame
    #         List of coordinates, if desired objects are present in frame
    #     """
    #     detectedObjects = self.result.boxes.cls.numpy().astype(int)
    #     foundObjects = np.where(detectedObjects == desiredObject)[0]
    #     if len(foundObjects) == 0:
    #         return False
    #     else:
    #         return self.result.boxes.xyxy.numpy()[foundObjects]
    #         # TODO: figure out how to return coordinates of all instances of specific object
    
    def checkCorrectItems(self, analysis: Result, neededItems: dict) -> dict:
        """ Determines what items (and how many) are missing from the current frame

        Args:
            neededItems (dict): Dictionary of items needed with quantities

        Returns:
            missingItems (dict): Dictionary of missing items and quantities left need for them
        """
        countDict = self.count(analysis)
        missingItems = dict()
        for item in neededItems:
            try:
                currentAmount = countDict[item]
                neededAmount = neededItems[item]
                if currentAmount < neededAmount:
                    missingItems[item] = neededAmount - currentAmount
            except:
                missingItems[item] = neededItems[item]
        
        return missingItems

    def checkEmpty(self, resultList: list):
        """Checks to see when the frame is empty of consumables. Will return a list where elements represent the frames where it is empty.

        Returns:
            frameStamps: List(int)

        """
        frameStamps = []
        for index, analysis in enumerate(resultList):
            if len(analysis.result.boxes.cls) == 0:
                frameStamps.append(index)

        return frameStamps

############################################################################

if __name__ == "__main__":
    testModel = ObjectDetectorYOLO(modelPath="runs/detect/train5/weights/best.pt")
    video = cv.VideoCapture(0)
    if not video.isOpened():
        print("Error opening camera")
    
    else:
        while True:
            ret, frame = video.read()
            timeMSEC = video.get(cv.CAP_PROP_POS_MSEC)
        
            if ret:
                testModel.detect(frame, timeStamp=timeMSEC)
                cv.imshow('frame', frame)
            
            if cv.waitKey(1) == ord('q'):
                break