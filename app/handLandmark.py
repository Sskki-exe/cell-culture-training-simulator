"""
Media Pipe Code is adapted from:
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python
"""
### Author: Edric Lay
### Date Created: 7/02/2025
############################################################################
### Library Imports
import mediapipe as mp
import cv2 as cv
import numpy as np
from result import Result
import camera
import time
from PIL import Image

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
############################################################################
class SanitisationMask():
    """
    Mask used to illustrate where sanisitation has occurred
    """
    def __init__(self, cameraProperties: list):
        cameraProperties[0] = int(cameraProperties[0])
        cameraProperties[1] = int(cameraProperties[1])
        self.cameraProperties = cameraProperties
        self.mask = np.zeros((cameraProperties[1],cameraProperties[0],3), dtype=np.uint8)
    
    def resetMask(self):
        """
        Resets the mask back to 0 (meaning need to resanitise)
        """
        self.mask = np.zeros((self.cameraProperties[1],self.cameraProperties[0],3), dtype=np.uint8)
    
    def calculateMaskCoverage(self):
        """ Determines how much of the mask has been 'painted'

        Returns:
            coverage (float): Percentage represenation of the mask that has been painted.
        """
        binaryMask = self.mask > 0 # Get a binary representation of mask
        rawCoverage=np.sum(np.sum(np.sum(binaryMask))) # Figure out how many '1's there are
        coverage = round(rawCoverage/(self.cameraProperties[1]*self.cameraProperties[0]) * 100,3) # Determine as percentage of total mask size
        return coverage
    
    def saveMaskImage(self, filename: str):
        copy = 255* np.ones_like(self.mask)
        img = Image.fromarray(cv.copyMakeBorder(copy-self.mask, 
                                                top=10, bottom=10, left=10, right=10, 
                                                borderType=cv.BORDER_CONSTANT, value=[0, 0, 0]))
        img.save(filename)


############################################################################

class handLandmarker():
    #############################
    # Values used for annotating on the image. Values taken from: 
    # https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    ############################
    def __init__(self, 
                 modelPath: str = "hand_landmarker.task", 
                 numHands: int = 2, 
                 minHandDetectionConfidence: float = 0.1, 
                 minHandPresenceConfidence: float = 0.1, 
                 minTrackingConfidence: float = 0.1, 
                 handLength: int = 20, 
                 cameraProperties: list = [640,480,30,cv.VideoWriter.fourcc(*"XVID")], 
                 backgroundDepth: int = 88):
        """
        Args:
            modelPath (str): path to the model being used
            numHands (int, optional): Maximum number of hands expected. Defaults to 2.
            minHandDetectionConfidence (float, optional): Minimum amount of confidence needed for a hand to be considered detected. Defaults to 0.1.
            minHandPresenceConfidence (float, optional): Minimum amount of confidence needed for a hand to still be considered present. Defaults to 0.1.
            minTrackingConfidence (float, optional): Minimum amount of confidence needed for the detector to keep tracking that hand, instead of re-detecting it. Defaults to 0.1.
            handLength (int, optional): The distance between the middle knuckle and the wrist in px. Defaults to 20            
            cameraProperties (list): Properties of camera being used in format = [width,height,fps,fourcc]
            backgroundDepth (int, optional): The distance to the background in cm. Defaults to 88
        """
        # Define options for detector
        options = mp.tasks.vision.HandLandmarkerOptions( 
            base_options = mp.tasks.BaseOptions(model_asset_path=modelPath),
            running_mode = mp.tasks.vision.RunningMode.VIDEO,
            num_hands = numHands,
            min_hand_detection_confidence = minHandDetectionConfidence,
            min_hand_presence_confidence = minHandPresenceConfidence,
            min_tracking_confidence = minTrackingConfidence
            # result_callback=self.updateResult
            )

        # Class Attributes
        self.result = mp.tasks.vision.HandLandmarkerResult # Store results in here
        self.result.handedness = []
        self.result.hand_landmarks = []
        self.result.hand_world_landmarks = []

        # Detector
        self.handLandmarker = mp.tasks.vision.HandLandmarker
        self.detector = self.handLandmarker.create_from_options(options) # Create detector here
        
        # Image Output Size
        self.cameraProperties = cameraProperties
        self.width = cameraProperties[0]
        self.height = cameraProperties[1]

        # Hand details
        self.backgroundDepth = backgroundDepth
        self.handLength = handLength
    
    ################################ Detection Function ############################################ 
    def detect(self, frame, timeStamp):
        """Method used to detect for hands in a frame

        Args
            frame: Frame in which we will be running the detector on
            timeStamp: Timestamp of frame
        
        Returns:
            result (Result): Result of the frame analyzed
        """
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) # Need to convert frame into mp.Image in order to use detect_async
        result=self.detector.detect_for_video(image = image, timestamp_ms = timeStamp)
        # status = len(result.handedness)
        return Result(timeStamp, result)

    ############################### Hand Location #############################################
    def updateHandRemovalFrame(self, resultList: list):
        """Updates the resultList frames to show handtracking based off a list of results

        Args:
            result (List(Result)): List of Result objects created after analysing a video 

        Returns:
            videoName (str): Name of the video with annotations
            count (int): Number of times a hand has been removed
        """
        prevHandCount = 0
        count = 0

        for analysis in resultList:
            annotated_image = self.draw_landmarks_on_image(analysis)
            analysis.frame = annotated_image
            curHandCount = len(analysis.result.handedness)
            if curHandCount < prevHandCount:
                count += 1
                analysis.status = False
            prevHandCount = curHandCount

        return count

    ############################### Sanitisation Coverage #############################################
    # def updateSanitisationFrames(self, result, cameraProperties: list):
    #     """Update the resultList frames to show the sanitisation on them

    #     Args:
    #         result (List(Result)): List of Result objects created after analysing a video 

    #     Returns:
    #         videoName (str): Name of the video with annotations
    #         coverage (float): Amount of the workspace which has been properly sanitised
    #     """
    #     mask = SanitisationMask(cameraProperties)

    #     for analysis in result:
    #         annotated_image = self.paintImage(analysis, mask)
    #         analysis.frame = annotated_image
    #         analysis.other = mask.calculateMaskCoverage()

    #     return mask.calculateMaskCoverage()
    
    def paintImage(self, frame, analysis: Result, mask: SanitisationMask):
        """Returns an image with paint representing areas a place has been santised

        Args:
            analysis (Result): Result object which is a frame which has been analysed
            mask (SanitisationMask): SanitisationMask object which tells how much of the space has been sanitised

        Returns:
            annotated_image (np.array): Image with 'paint' on it to demonstrate sanitisation

        """
        annotated_image = np.copy(frame) 
        detection_result = analysis.result
        handsDetected = analysis.result.handedness
        if handsDetected:
            hand_landmarks_list = detection_result.hand_landmarks
            
            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]

                x,y,z = self.getNormalMiddleKnuckle(hand_landmarks) # Get coordinate of middle of hand
                
                try:
                    depth = self.getDepth(hand_landmarks) # Get the depth of the hand(s) locations
                    annotated_image = self.drawCentreCircle(annotated_image, [x, y, z])
                    # print(f"Depth: {depth}")

                    if depth > self.backgroundDepth: 
                        (circleCentre) = self.convertNormalToImageCoord(x,y) # Convert the normal coordinate representation of the hand into image coordinates
                        middleFingerTip = np.array(self.convertNormalToImageCoord(hand_landmarks[12].x,hand_landmarks[12].y)) # Gets the coordinate of the middle knuckle
                        wrist = np.array(self.convertNormalToImageCoord(hand_landmarks[0].x,hand_landmarks[0].y)) # Gets the coordinate of the wrist
                        circleRadius = int(np.linalg.norm(middleFingerTip-wrist)/2) # Assume sanitation is circular, with origin on the middle knuckle and diameter the length of the wrist to the middle tip finger
                        # print(f"Circle Radius: {circleRadius}")
                        try:
                            cv.circle(mask.mask, center=circleCentre, radius=circleRadius, color=(0, 255, 0), thickness=cv.FILLED )# Draw the circle around the centre knuckle on the mask

                        except:
                            pass # if errors, don't paint!!!!
                except:
                    pass
                

        # Add mask to original image to create santisation of image
        opacity = 0.3
        paintedImage = cv.addWeighted(annotated_image, 1 - opacity, mask.mask, opacity, 0) 
        annotated_image = np.where(mask.mask > 0, paintedImage, annotated_image)


        cv.putText(annotated_image, f"Analysing Sanitisation of Workspace",
                    (0, 25), cv.FONT_HERSHEY_DUPLEX,
                    handLandmarker.FONT_SIZE/2, (0,0,0), handLandmarker.FONT_THICKNESS, cv.LINE_AA)

        cv.putText(annotated_image, f"Percentage Sanitised: {round(mask.calculateMaskCoverage(),2)}%",
                    (0, 50), cv.FONT_HERSHEY_DUPLEX,
                    handLandmarker.FONT_SIZE/2, (0,0,0), handLandmarker.FONT_THICKNESS, cv.LINE_AA)

        return annotated_image

    ############################### Coordinate Calculations #############################################
    def convertNormalToImageCoord(self,normalX,normalY):
        """Converts normal coordinates to image coordinates

        Args:
            normalX (float): Normalised position of x
            normalY (float): Normalised position of y

        Returns
            coordinates (x,y): Pixel coordinates in image coordinates
        """
        return solutions.drawing_utils._normalized_to_pixel_coordinates(normalX,normalY, self.width, self.height)

    def getDepth(self, detection_result) -> float:
        """Returns the distance of a hand from the camera

        Args:
            detection_result (list): List of landmarks on a hand in the frame

        Returns:
            depth (float): Distance from camera to hand 
        """
        middleKnuckle = np.array(self.convertNormalToImageCoord(detection_result[9].x,detection_result[9].y)) # Gets the coordinate of the middle knuckle
        wrist = np.array(self.convertNormalToImageCoord(detection_result[0].x,detection_result[0].y)) # Gets the coordinate of the wrist
        pixelDist = np.linalg.norm(middleKnuckle-wrist) # Figure out the amount of pixels used
        # print(pixelDist)
        depth = self.backgroundDepth * self.handLength/pixelDist # Use similar triangles to figure out the depth (very primitive method)
        return depth
        
    def getNormalMiddleKnuckle(self, detection_result):
        """Gets the normalized coordinate of middleKnuckle
        Args:
            detection_result (list): List of handlandmarks for one hand
        Returns:
            x,y,z: Normalized coordinates of the middle knuckle
        """
        x,y,z = detection_result[9].x,detection_result[9].y,detection_result[9].y
        return x,y,z
    
    ############################### Drawing Tools #############################################
    def draw_landmarks_on_image(self, frame, analysis: Result):
        """ Function which will draw all 15 hand landmarks on the frame. Adapted from: https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1.
        
        Args:
            analysis (Result): Result object which is a frame which has been analysed

        Returns:
            annotated_image (np.array): annotated_image
        """
        annotated_image = np.copy(frame) 
        detection_result = analysis.result
        handsDetected = len(analysis.result.handedness)

        if handsDetected > 0: # The following code was taken from above link
            hand_landmarks_list = detection_result.hand_landmarks
            handedness_list = detection_result.handedness

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                handedness = handedness_list[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())

                # # Get the top left corner of the detected hand's bounding box.
                # height, width, _ = annotated_image.shape
                # x_coordinates = [landmark.x for landmark in hand_landmarks]
                # y_coordinates = [landmark.y for landmark in hand_landmarks]
                # text_x = int(min(x_coordinates) * width)
                # text_y = int(min(y_coordinates) * height) - handLandmarker.MARGIN

                # # Draw handedness (left or right hand) and cofindence on the image.
                # cv.putText(annotated_image, f"Hand: {handedness[0].category_name}",
                #             (text_x, text_y - 25), cv.FONT_HERSHEY_DUPLEX,
                #             handLandmarker.FONT_SIZE, handLandmarker.HANDEDNESS_TEXT_COLOR, handLandmarker.FONT_THICKNESS, cv.LINE_AA)
                # cv.putText(annotated_image, f"Confidence Level: {round(handedness[0].score, 3)}",
                #     (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                #     handLandmarker.FONT_SIZE, handLandmarker.HANDEDNESS_TEXT_COLOR, handLandmarker.FONT_THICKNESS, cv.LINE_AA)
            cv.putText(annotated_image, "Hand Detection: Marking hand landmarks",
            (0, 25), cv.FONT_HERSHEY_DUPLEX,
            handLandmarker.FONT_SIZE/2, (0,0,0), handLandmarker.FONT_THICKNESS, cv.LINE_AA)

        elif analysis.status:
            cv.putText(annotated_image, "Hand Detection: A hand was removed!",
            (0, 25), cv.FONT_HERSHEY_DUPLEX,
            handLandmarker.FONT_SIZE/2, (0,0,0), handLandmarker.FONT_THICKNESS, cv.LINE_AA)

        else:
            cv.putText(annotated_image, "Hand Detection: No hands have been detected!",
            (0, 25), cv.FONT_HERSHEY_DUPLEX,
            handLandmarker.FONT_SIZE/2, (0,0,0), handLandmarker.FONT_THICKNESS, cv.LINE_AA)
        
        return annotated_image

    def drawCentreCircle(self, rgb_image, middleKnuckle):
        """
        Draws the landmark that represents the centre of hand on middle knuckle. Adapted from: https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb#scrollTo=s3E6NFV-00Qt&uniqifier=1.

        Args:
            rgb_image (np.array): Frame to be annotated
            middleKnuckle (list): Coordinates of middle knuckle

        Returns:
            annotated_image (np.array): Image with centre knuckle of hand marked
        """
        annotated_image = np.copy(rgb_image)
        
        # Draw the hand landmark.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=middleKnuckle[0], y=middleKnuckle[1], z=middleKnuckle[2])
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
        )

        return annotated_image
    
    ##############################Result Smoother#############################################  

    def checkHandRemoved(self, resultList: list) -> list:
        """Checks to see what frame's have a hand that has been removed and return boolean list.
        
        Args:
            resultList (list): List of Result objects
            filterRange (int): Range of n samples to include when comparing e.g [x-n:x+n]
        
        Returns:
            booleanList (list(tuples)): List of boolean tuples where each element represents if there are less hands then the surrounding frames. Check function code for more description
        """
        handsList = [len(analysis.result.handedness) for analysis in resultList] # Create list of hand count for each result
        
        # Create list representation of previous handcounts
        prevHandsList = handsList[:-2]
        prevHandsList = np.array(prevHandsList)

        # Create list represenation for future handcounts
        futureHandsList = handsList[2:]
        futureHandsList = np.array(futureHandsList)

        # Vector comparison
        currentHandsList = np.array(handsList[1:-1]) # Do not include first and last frame for ease
        
        """
        An element i is only True in the boolean list when:
        
        - If prev[i]==future[i] (Condition 1), then True if prev[i]==current[i]==future[i], otherwise False (Condition A). (True, x) 

        - Elif prev[i]!=future[1] (Condition 1), then True if prev[i]==current[i] or current[i]=future[i], otherwise False (Condition B). (False, x) (2)

        """
        # booleanList = np.where(prevHandsList==futureHandsList, (True, (prevHandsList==currentHandsList) & (currentHandsList==futureHandsList)), (False, (prevHandsList==currentHandsList)|(currentHandsList==futureHandsList)))

        # State 1:
        cond1 = prevHandsList==futureHandsList
        condA = (prevHandsList==currentHandsList) & (currentHandsList==futureHandsList)
        condB = (prevHandsList==currentHandsList)|(currentHandsList==futureHandsList)

        booleanList = np.column_stack((
            np.where(cond1, True, False),
            np.where(cond1, condA, condB)
            ))
        
        return booleanList.tolist()

    def smoothHandLandmarks(self, resultList: list):
        """If there is a no detection frame surrounded by multiple frames with a detection, it can be reasonably determined that there has been a false negative occurance. 

        Args:
            resultList (list): List of results

        Returns:
            smoothedResultList (list): List of smoothed results
        """
        handsChangeList = self.checkHandRemoved(resultList) # Check if results have False Negatives
        smoothedResultList = resultList[:1]

        # Loop through all items in the resultList
        for i in range(len(handsChangeList)):
            if handsChangeList[i][1]: # If no issues, nothing needed to be done
                smoothedResultList.append(resultList[i+1])
            
            else:
                if handsChangeList[i][0]: # Intermediate frame does not match surrounding frames
                    currentResult = resultList[i+1]
                    timeStamp = currentResult.time
                    prevResult = resultList[i].result
                    futureResult = resultList[i+2].result

                    # Make assumption that during false negative, hand was into the middle of the prev and future locations
                    averageResult = self.averageHandLandmarkResults(prevResult,futureResult)

                    smoothedResult = Result(timeStamp,averageResult)
                    smoothedResultList.append(smoothedResult)

                else: # Only include the additional hand between the two of them when it has above 50% confidence (Can only trigger if (1,0,2) or (2,0,1) I believe)
                    currentResult = resultList[i+1]
                    prevResult = resultList[i].result
                    futureResult = resultList[i+2].result

                    if len(prevResult.handedness) > len(futureResult.handedness):
                        takeFuture = False
                        for hand in prevResult.handedness:
                            if hand[0].score < 0.5:
                                takeFuture = True
                                break
                        
                        if takeFuture:
                            currentResult.result = futureResult
                        
                        else:
                            currentResult.result = prevResult
                    
                    else: # len(prevResult.handedness) < len(futureResult.handedness)
                        takePrev = False
                        for hand in futureResult.handedness:
                            if hand[0].score < 0.5:
                                takePrev = True
                                break
                        
                        if not takePrev:
                            currentResult.result = futureResult
                        
                        else:
                            currentResult.result = prevResult

                    smoothedResultList.append(currentResult)
        
        smoothedResultList.append(resultList[-1])

        return smoothedResultList

    def averageHandLandmarkResults(self, prevResult, futureResult):
        """Finds the average result of the two frames

        Args:
            prevResult (mp.tasks.vision.HandLandmarkerResult): Previous result of frames
            futureResult (mp.tasks.vision.HandLandmarkerResult): Future result of frames

        Returns:
            (mp.tasks.vision.HandLandmarkerResult): Average result of the two frames
        """
        # Find out which element in each list belongs to which hand
        prevLeft = False
        prevRight = False
        futureLeft = False
        futureRight = False

        for index, hand in enumerate(prevResult.handedness): # Can assume only one user, so max two hands (one left, one right)
            if hand[0].category_name == "Left":
                prevLeft = index
            elif hand[0].category_name == "Right":
                prevRight = index

        for index, hand in enumerate(futureResult.handedness):
            if hand[0].category_name == "Left":
                futureLeft = index
            elif hand[0].category_name == "Right":
                futureRight = index


        # Find average hand landmarks. Note: We are assuming that the frames don't change hand type
        averagedHandLandmarks = [] #
        if type(prevLeft) == bool:
            pass
        else:
            prevLandmark = prevResult.hand_landmarks[prevLeft]
            futureLandmark = futureResult.hand_landmarks[futureLeft]

            averageLandmarks = []
            for prevCoord, futureCoord in zip(prevLandmark,futureLandmark):
                aveX = (prevCoord.x + futureCoord.x)/2
                aveY = (prevCoord.y + futureCoord.y)/2
                aveZ = (prevCoord.z + futureCoord.z)/2
                averageLandmarks.append(landmark_pb2.NormalizedLandmark(x=aveX, y=aveY, z=aveZ))
            
            averagedHandLandmarks.insert(prevLeft,averageLandmarks)
        
        if type(prevRight) == bool:
            pass
        else:
            prevLandmark = prevResult.hand_landmarks[prevRight]
            futureLandmark = futureResult.hand_landmarks[futureRight]

            averageLandmarks = []
            for prevCoord, futureCoord in zip(prevLandmark,futureLandmark):
                aveX = (prevCoord.x + futureCoord.x)/2
                aveY = (prevCoord.y + futureCoord.y)/2
                aveZ = (prevCoord.z + futureCoord.z)/2
                averageLandmarks.append(landmark_pb2.NormalizedLandmark(x=aveX, y=aveY, z=aveZ))
            
            averagedHandLandmarks.insert(prevRight,averageLandmarks)

        # Find average hand world landmarks
        averagedHandWorldLandmarks = [] #
        if type(prevLeft) == bool:
            pass
        else:
            prevLandmark = prevResult.hand_world_landmarks[prevLeft]
            futureLandmark = futureResult.hand_world_landmarks[futureLeft]

            averageLandmarks = []
            for prevCoord, futureCoord in zip(prevLandmark,futureLandmark):
                aveX = (prevCoord.x + futureCoord.x)/2
                aveY = (prevCoord.y + futureCoord.y)/2
                aveZ = (prevCoord.z + futureCoord.z)/2
                averageLandmarks.append(landmark_pb2.NormalizedLandmark(x=aveX, y=aveY, z=aveZ))
            
            averagedHandWorldLandmarks.insert(prevLeft,averageLandmarks)
        
        if type(prevRight) == bool:
            pass
        else:
            prevLandmark = prevResult.hand_world_landmarks[prevRight]
            futureLandmark = futureResult.hand_world_landmarks[futureRight]

            averageLandmarks = []
            for prevCoord, futureCoord in zip(prevLandmark,futureLandmark):
                aveX = (prevCoord.x + futureCoord.x)/2
                aveY = (prevCoord.y + futureCoord.y)/2
                aveZ = (prevCoord.z + futureCoord.z)/2
                averageLandmarks.append(landmark_pb2.NormalizedLandmark(x=aveX, y=aveY, z=aveZ))
            
            averagedHandWorldLandmarks.insert(prevRight,averageLandmarks)

        return mp.tasks.vision.HandLandmarkerResult(
            handedness=prevResult.handedness, # Handedness is the same as both frames
            hand_landmarks=averagedHandLandmarks,
            hand_world_landmarks=averagedHandWorldLandmarks
        )
    
    def getHandSpeed(self, prevResult, currentResult):
        prevLeft = False
        prevRight = False
        currentLeft = False
        currentRight = False
        speed = 0
        speedx = 0 # Assume base line speed

        for index, hand in enumerate(prevResult.handedness): # Can assume only one user, so max two hands (one left, one right)
            if hand[0].category_name == "Left":
                prevLeft = index
            elif hand[0].category_name == "Right":
                prevRight = index

        for index, hand in enumerate(currentResult.handedness):
            if hand[0].category_name == "Left":
                currentLeft = index
            elif hand[0].category_name == "Right":
                currentRight = index

        # # With intrinsic matrix
        # if type(prevLeft)==int and type(prevLeft) == type(currentLeft): 
        #     # can compare speed in left hand
        #     worldPrev = prevResult.hand_landmarks[prevLeft][0]
        #     worldPrev = self.convertNormalToImageCoord(worldPrev.x, worldPrev.y)
        #     prevDepth = self.getDepth(prevResult.hand_landmarks[prevLeft])
        #     prevWorld = prevDepth * np.linalg.inv(camera.K) @ np.array([[worldPrev[0]],[worldPrev[1]],[1]])

        #     worldCurrent = currentResult.hand_landmarks[currentLeft][0]
        #     worldCurrent = self.convertNormalToImageCoord(worldCurrent.x, worldCurrent.y)
        #     currentDepth = self.getDepth(currentResult.hand_landmarks[currentLeft])
        #     currentWorld = currentDepth * np.linalg.inv(camera.K) @ np.array([[worldCurrent[0]],[worldCurrent[1]],[1]])

        #     leftDist = np.linalg.norm(currentWorld-prevWorld)
        #     leftspeed = leftDist*30

        # else:
        #     leftspeed = 0

        # if type(prevRight)==int and type(prevRight) == type(currentRight): 
        #     # can compare speed in right hand
        #     worldPrev = prevResult.hand_landmarks[prevRight][0]
        #     worldPrev = self.convertNormalToImageCoord(worldPrev.x, worldPrev.y)
        #     prevDepth = self.getDepth(prevResult.hand_landmarks[prevRight])
        #     prevWorld = prevDepth * np.linalg.inv(camera.K) @ np.array([[worldPrev[0]],[worldPrev[1]],[1]])

        #     worldCurrent = currentResult.hand_landmarks[currentRight][0]
        #     worldCurrent = self.convertNormalToImageCoord(worldCurrent.x, worldCurrent.y)
        #     currentDepth = self.getDepth(currentResult.hand_landmarks[currentRight])
        #     currentWorld = currentDepth * np.linalg.inv(camera.K) @ np.array([[worldCurrent[0]],[worldCurrent[1]],[1]])

        #     print(prevWorld, currentWorld)

        #     rightDist = np.linalg.norm(currentWorld-prevWorld)
        #     rightspeed = rightDist*30
        
        # else:
        #     rightspeed = 0

        # speed = max(leftspeed, rightspeed)

        # With world coordinates

        if type(prevLeft)==int and type(prevLeft) == type(currentLeft): 
            # can compare speed in left hand
            worldPrev = prevResult.hand_world_landmarks[prevLeft][0]
            worldPrev = np.array([worldPrev.x,worldPrev.y,worldPrev.z])

            worldCurrent = currentResult.hand_world_landmarks[currentLeft][0]
            worldCurrent = np.array([worldCurrent.x,worldCurrent.y,worldCurrent.z])

            leftDist = np.linalg.norm(worldPrev-worldCurrent)
            leftxspeed = leftDist*30

        else:
            leftxspeed = 0

        if type(prevRight)==int and type(prevRight) == type(currentRight): 
            # can compare speed in right hand
            worldPrev = prevResult.hand_world_landmarks[prevRight][0]
            worldPrev = np.array([worldPrev.x,worldPrev.y,worldPrev.z])

            worldCurrent = currentResult.hand_world_landmarks[currentRight][0]
            worldCurrent = np.array([worldCurrent.x,worldCurrent.y,worldCurrent.z])

            rightDist = np.linalg.norm(worldPrev-worldCurrent)
            rightxspeed = rightDist*30
        
        else:
            rightxspeed = 0

        speedx = max(leftxspeed, rightxspeed)*100 # into cm/s

        return speedx
    
    ################################ Archaic Functions ############################################ 
    # def analyseVideo(self, videoName: str):
    #     """Perform a handtracking analysis of a full video

    #     Args:
    #         videoName (str): File name of the video

    #     Returns:
    #         videoAnalysis (list): List of results of each frame analyzed
    #     """
    #     footage = cv.VideoCapture(videoName)
    #     videoAnalysis = []

    #     while footage.isOpened():
    #         try:
    #             ret, frame = footage.read()
    #             if ret:
    #                 timeStamp = footage.get(cv.CAP_PROP_POS_MSEC)
    #                 frameResult = self.detect(frame, timeStamp)
    #                 videoAnalysis.append(frameResult)

    #             if ret == False or cv.waitKey(1) == ord('q'):
    #                 break
    #         except Exception as e:
    #             print(f"footage error: {e}")

    #     footage.release()
    #     return videoAnalysis

    # def updateResult(self, result, output_image: mp.Image, timestamp_ms: int):
    #     """Callback function used by options when detector is run. 

    #     Args:
    #         result (mp.tasks.vision.HandLandmarkerResult) 
    #         output_image (mp.Image)
    #         timestamp_ms (int)
    #     """
    #     self.result = result


if __name__ == "__main__":
    # Get Camera Properties
    cap = cv.VideoCapture(0)
    cameraProperties = camera.getCameraProperties(cap)

    testMask = SanitisationMask(cameraProperties)
    handDetector = handLandmarker(modelPath="hand_landmarker.task", cameraProperties = cameraProperties)
    count = 0
    ret, frame = cap.read()
    currentResult = handDetector.detect(frame, count)


    while True:
        ret, frame = cap.read()
        count += 1
        if ret:
            prevResult = currentResult
            currentResult = handDetector.detect(frame, count)
            cv.imshow("frame", frame)
            speed = handDetector.getHandSpeed(prevResult.result, currentResult.result)

            if cv.waitKey(1) == ord("q"):
                break
            
            count += 1
            
    cap.release()