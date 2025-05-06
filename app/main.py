### Author: Edric Lay
### Date Created: 22/04/2025
############################################################################
import customtkinter as ctk
import tkinter as tk
import cv2 as cv
import camera
import result as R
from objectDetectionYOLO import ObjectDetectorYOLO
from handLandmark import handLandmarker, SanitisationMask
from PIL import Image, ImageTk
import time
from datetime import datetime
import os
from serial_read_sample import read_hub_serial
import csv

def convertCVtoPIL(frame):
    """Function to convert from openCV array to PIL array

    Args:
        frame (np.array): Frame captured by openCV 

    Returns:
        frame (PIL.Image): Frame captured by openCV in PIL format
    """
    RGBframe = cv.cvtColor(frame, cv.COLOR_BGR2RGB) # cv use BGR and PIL uses RGB, so need to convert 
    RGBframe = Image.fromarray(RGBframe)
    RGBframe = ImageTk.PhotoImage(image=RGBframe)
    return RGBframe

class app(ctk.CTk):
    """Application used to control the system
    """
    def __init__(self):
        super().__init__()
        
        # Display Initialisation
        self.title("Training Cell Culture Skills")
        self.geometry("1920x1080")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure(0, weight=1)

        # Functionality
        self.state('zoomed') # Full screen
        self.bind("<Escape>", lambda x: self.destroy()) # Kills app when hit escape

        # Loading Screen 1
        self.display = loadingFrame(self, loadingImageStr="loadingGraphics/1.JPG")
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()

        # Camera Initialisation
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened:
            print("Camera brokey")
            exit()

        # Loading Screen 2
        self.display = loadingFrame(self, loadingImageStr="loadingGraphics/2.JPG")
        self.update()

        # Change camera properties to be as best as they can be for more accurate detection
        self.cameraProperties = camera.getCameraProperties(self.cap)

        # Loading Screen 2
        self.display = loadingFrame(self, loadingImageStr="loadingGraphics/3.JPG")
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()

        # Model Initalisation
        self.handDetector = handLandmarker(cameraProperties=self.cameraProperties) # Note: might not need to be restarted
        self.objectDetector = ObjectDetectorYOLO()
        self.function = 0

        # Results interpreted from most recent recording
        self.videoName = [] 

        # Show loading menu
        self.display = menuFrame(self)
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()

    def displayMenu(self):
        """Function to display menu
        """
        self.handDetector = handLandmarker(cameraProperties=self.cameraProperties) # Note: might not need to be restarted
        self.display = menuFrame(self)
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()
    
    def displaySingleChannel(self):
        """Function to record screen for sanitisation
        """
        self.tool = 0
        self.display = recordScreen(self, 0)
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()

    def displayPipetteAid(self):
        """Function to record screen for object detection
        """
        self.tool = 1
        self.display = recordScreen(self, 1)
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()

    def displayProcessVideo(self, usage: str, dateStr):
        """Function to playback video with annotations

        Args:
            usage (str): One word description of what the recording was about
        """
        self.display = processVideoScreen(self, usage, dateStr)
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()
    
    def close(self):
        """Function to destroy app from menu screen
        """
        self.cap.release()
        self.destroy()

class menuFrame(ctk.CTkFrame):
    """Frame used as the application menu
    """
    def __init__(self, master, fg_color = None):
        super().__init__(master, width = 2000, height = 2000)

        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure(0, weight = 1)
        self.grid_rowconfigure(1, weight = 0, uniform="cell") # Just do sanitization
        self.grid_rowconfigure(2, weight = 0, uniform="cell") # Just do object identification/verification 
        self.grid_rowconfigure(3, weight = 0, uniform="cell") # Do a full run
        self.grid_rowconfigure(4, weight = 0, uniform="cell") # Close app

        
        self.titleLabel = ctk.CTkLabel(self, text = "Cell Culture Training", font = ("Segoe UI", 100, "bold"))
        self.titleLabel.grid(row = 0, column = 0, pady=10, padx=10, sticky = "nsew")

        self.singleChannelPippetteButton = ctk.CTkButton(self, text = "Assess Single Channel Pippetting", command = master.displaySingleChannel)
        self.singleChannelPippetteButton.grid(row = 1, column = 0, pady=10, padx=10, sticky = "nsew")

        self.pippetteAidButton = ctk.CTkButton(self, text = "Assess Pippetoir Skills", command = master.displayPipetteAid)
        self.pippetteAidButton.grid(row = 2, column = 0, pady=10, padx=10, sticky = "nsew")
         
        self.escapeButton = ctk.CTkButton(self, text = "Exit", command = master.close, fg_color="red", hover_color="red")
        self.escapeButton.grid(row = 3, column = 0, pady=10, padx=10, sticky = "nsew")

class loadingFrame(ctk.CTkFrame):
    """Frame used for loading screen
    """
    def __init__(self, master, loadingImageStr: str = None):
        super().__init__(master, width = 2000, height = 2000)

        # Open poster image once and calculate ratio
        with Image.open(loadingImageStr) as loadingImage:
            loadingImage = ctk.CTkImage(
                light_image=loadingImage.copy(),
                dark_image=loadingImage.copy(),
                size=(loadingImage.width, loadingImage.height)
            )
        
        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure(0, weight = 1)

        self.loadImage = ctk.CTkLabel(self, image=loadingImage, text="")
        self.loadImage.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")

class recordScreen(ctk.CTkFrame):
    """Frame used to record webcam footage
    """
    def __init__(self, master, usage: int):
        super().__init__(master)
        self.master = master
        self.recordCamera = False
        
        if usage == 0:
            self.no = 0
            self.usage = 'Single Channel Pipettor'

        elif usage == 1:
            self.no = 1
            self.usage = 'Pipette Controller'

        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 1) 
        self.grid_rowconfigure(2, weight = 0) 
        
        ### PUT PLACEHOLDER IMAGE WHILST CAMERA IS OFF
        # self.placeholder = ctk.CTkImage(light_image=Image.open("loadingGraphics/wait.jpg"),
        #                     dark_image=Image.open("loadingGraphics/wait.jpg"),
        #                     size=(1280, 720))
        
        self.titleLabel = ctk.CTkLabel(self, text = f"Recording", font = ("Segoe UI", 100, "bold"))
        self.titleLabel.grid(row = 0, column = 0, pady=10, padx=10, sticky = "nsew")

        self.cameraFrame = tk.Canvas(self, width = self.master.cameraProperties[0], height = self.master.cameraProperties[1], highlightthickness=1)
        self.cameraFrame.grid(row = 1, column = 0, pady = 10, padx = 10)
        
        # Display Placeholder
        tempFrame = Image.open("loadingGraphics/wait.jpg")
        tempFrame = ImageTk.PhotoImage(image=tempFrame)

        self.displayImage = self.cameraFrame.create_image(0, 0, anchor=tk.NW, image=tempFrame)
        self.cameraFrame.image = tempFrame

        self.recordButton = ctk.CTkButton(self, text = "Start Recording", command = self.recordSession)
        self.recordButton.grid(row = 2, column = 0, pady=10, padx=10, sticky = "nsew")

    def recordSession(self):
        """Function used to record all the video for a videoWriter object

        Args:
            videoWriter (cv.VideoWriter): VideoWriter object for video to be written to
            cap (cv.VideoCapture): Camera being used to capture video
        
        """

        # Make Directory to save videos
        date = datetime.now()
        dateStr = date.strftime("%d%m%y %H%M%S")
        try:
            os.makedirs(f"video/{dateStr}")
        except:
            pass
    
        try:
            os.makedirs(f"video/{dateStr}/final")
        except:
            pass

        try:
            os.makedirs(f"video/{dateStr}/raw")
        except:
            pass

        try:
            os.makedirs(f"video/{dateStr}/process")
        except:
            pass

        # Step 1: Beginning Sanitisation
        videoWriter, videoName = camera.createVideoWriter(function=f"{dateStr}/raw/sanitise", cameraProperties=self.master.cameraProperties)
        self.recordButton.configure(command = self.endVideo)
        self.recordButton.configure(text = "Start Collection")
        self.update()

        if not videoWriter.isOpened():
            print("AHHHHHHHHHHHH")
            exit()

        print("Start recording sanisation")
        self.recordCamera = True

        start = time.time()

        count = 0
        while self.recordCamera == True:
            ret, frame = self.master.cap.read()
            if ret:
                videoWriter.write(frame)

                framePIL = convertCVtoPIL(frame)
                
                self.cameraFrame.create_image(0, 0, anchor=tk.NW, image=framePIL)
                self.cameraFrame.image = framePIL
                self.update()
                # time.sleep(1/30)
                count += 1

            if cv.waitKey(1) == ord('q'):
                break
        
        end = time.time()
        print(count/(end-start))

        videoWriter.release()
        self.master.videoName.append(videoName)
        
        # Step 2: Collect Materials
        videoWriter, videoName = camera.createVideoWriter(function=f"{dateStr}/raw/collection", cameraProperties=self.master.cameraProperties)
        self.recordButton.configure(command = self.endVideo)
        self.recordButton.configure(text = f"Start Using {self.usage}")
        self.update()

        if not videoWriter.isOpened():
            print("AHHHHHHHHHHHH")
            exit()

        print("Start recording collection")
        self.recordCamera = True
        
        start = time.time()

        count = 0
        while self.recordCamera == True:
            ret, frame = self.master.cap.read()
            if ret:
                videoWriter.write(frame)

                framePIL = convertCVtoPIL(frame)
                
                self.cameraFrame.create_image(0, 0, anchor=tk.NW, image=framePIL)
                self.cameraFrame.image = framePIL
                self.update()
                # time.sleep(1/30)
                count += 1

            if cv.waitKey(1) == ord('q'):
                break
        
        end = time.time()
        print(count/(end-start))

        videoWriter.release()
        self.master.videoName.append(videoName)
  

        # Step 3: Use Tool
        videoWriter, videoName = camera.createVideoWriter(function=f"{dateStr}/raw/tool", cameraProperties=self.master.cameraProperties)
        self.recordButton.configure(command = self.endVideo)
        self.recordButton.configure(text = f"Start cleaning space")
        self.update()

        if not videoWriter.isOpened():
            print("AHHHHHHHHHHHH")
            exit()

        file = open(f"{dateStr}/raw/data.csv", mode='w', newline='')
        writer = csv.DictWriter(file, fieldnames=["RollF","PitchF","Button"])
        
        def serial2Dict(serialString: str):
            """Function to convert from the string outputted from the Arduino into a dictionary for CSV saving.

            Args:
                serialString (str): String from Arduino

            Returns:
                dict: Dictionary explaining what string was
            """
            label, data = serialString.split(":")
            roll, pitch, button = data.split("/")
            row = {
                "RollF":roll,
                "PitchF":pitch,
                "Button":button
            }
            return row


        print("Start recording tool usage")
        self.recordCamera = True

        start = time.time()

        count = 0
        while self.recordCamera == True:
            ret, frame = self.master.cap.read()
            if ret:
                videoWriter.write(frame)
                
                framePIL = convertCVtoPIL(frame)
                
                self.cameraFrame.create_image(0, 0, anchor=tk.NW, image=framePIL)
                self.cameraFrame.image = framePIL
                string = read_hub_serial()

                strDict = serial2Dict(string)

                writer.writerow(strDict)
                self.update()
                # time.sleep(1/30)
                count += 1

            if cv.waitKey(1) == ord('q'):
                break
        
        end = time.time()
        print(count/(end-start))

        videoWriter.release()
        self.master.videoName.append(videoName)

        # Step 4: Clean Space
        videoWriter, videoName = camera.createVideoWriter(function=f"{dateStr}/raw/disposal", cameraProperties=self.master.cameraProperties)
        self.recordButton.configure(command = self.endVideo)
        self.recordButton.configure(text = f"Start sanitising space")
        self.update()

        if not videoWriter.isOpened():
            print("AHHHHHHHHHHHH")
            exit()

        print("Start recording clean up")
        self.recordCamera = True

        start = time.time()

        count = 0
        while self.recordCamera == True:
            ret, frame = self.master.cap.read()
            if ret:
                videoWriter.write(frame)

                framePIL = convertCVtoPIL(frame)
                
                self.cameraFrame.create_image(0, 0, anchor=tk.NW, image=framePIL)
                self.cameraFrame.image = framePIL
                self.update()
                # time.sleep(1/30)
                count += 1

            if cv.waitKey(1) == ord('q'):
                break
        
        end = time.time()
        print(count/(end-start))

        videoWriter.release()
        self.master.videoName.append(videoName)

        # Step 5: Sanitise Space
        videoWriter, videoName = camera.createVideoWriter(function=f"{dateStr}/raw/sanitise", cameraProperties=self.master.cameraProperties)
        self.recordButton.configure(command = self.endVideo)
        self.recordButton.configure(text = f"End Video")
        self.update()

        if not videoWriter.isOpened():
            print("AHHHHHHHHHHHH")
            exit()

        print("Start recording sanisation")
        self.recordCamera = True

        start = time.time()

        count = 0
        while self.recordCamera == True:
            ret, frame = self.master.cap.read()
            if ret:
                videoWriter.write(frame)

                framePIL = convertCVtoPIL(frame)
                
                self.cameraFrame.create_image(0, 0, anchor=tk.NW, image=framePIL)
                self.cameraFrame.image = framePIL
                self.update()
                # time.sleep(1/30)
                count += 1

            if cv.waitKey(1) == ord('q'):
                break
        
        end = time.time()
        print(count/(end-start))

        videoWriter.release()
        self.master.videoName.append(videoName)
        self.master.displayProcessVideo(self.no, dateStr)
        self.destroy()

    def endVideo(self):
        """Function to end the recording of the video, as triggered by a button.
        """
        self.recordCamera = False
        # self.cameraFrame = ctk.CTkLabel(self, fg_color="transparent", image = self.placeholder, text = "")
        # self.cameraFrame.grid(row = 1, column = 0, pady = 10, padx = 10, sticky = "nsew")
        self.update()


class processVideoScreen(ctk.CTkFrame):
    """Frame to play video after processing and analysis
    """
    def __init__(self, master, usage: int, dateStr: str):
        super().__init__(master)
        self.master = master
        self.usage = usage
        self.dateStr = dateStr

        # Statistics
        self.startSanitisation = 0
        self.handRemovals = 0
        self.correctMaterials = False
        self.emptySpace = False
        self.endSanitisation = 0

        # Graphics
        self.grid_columnconfigure(0, weight = 1)
        self.grid_columnconfigure(1, weight = 0)
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 1) 
        self.grid_rowconfigure(2, weight = 0) 
    
        ### PUT PLACEHOLDER IMAGE WHILST CAMERA IS OFF
        self.placeholder = ctk.CTkImage(light_image=Image.open("loadingGraphics/wait.jpg"),
                            dark_image=Image.open("loadingGraphics/wait.jpg"),
                            size=(1280, 720))
        
        self.titleLabel = ctk.CTkLabel(self, text = f"Reviewing", font = ("Segoe UI", 100, "bold"))
        self.titleLabel.grid(row = 0, column = 0, columnspan = 2, pady=10, padx=10, sticky = "nsew")

        self.cameraFrame = tk.Canvas(self, width = self.master.cameraProperties[0], height = self.master.cameraProperties[1], highlightthickness=1)
        self.cameraFrame.grid(row = 1, column = 0, pady = 10, padx = 10)

        tempFrame = Image.open("loadingGraphics/wait.jpg")
        tempFrame = ImageTk.PhotoImage(image=tempFrame)

        self.displayImage = self.cameraFrame.create_image(0, 0, anchor=tk.NW, image=tempFrame)
        self.cameraFrame.image = tempFrame

        self.playButton = ctk.CTkButton(self, text = "Process video", command = self.processVideo)
        self.playButton.grid(row = 2, columnspan = 2, column = 0, pady=10, padx=10, sticky = "nsew")

        self.statistics = ctk.CTkFrame(self)
        self.statistics.grid(row = 1, column = 1, pady=10, padx=10, sticky = "nsew")

        # Statistics frame
        self.statisticsCoverage = ctk.CTkLabel(self.statistics, text = f"Percentage workspace sanitised: {self.startSanitisation}%", font = ("Segoe UI", 10, "bold"))
        self.statisticsCoverage.grid(row = 0, column = 0, pady=10, padx=10, sticky = "nsew")

        self.statisticsHandRemoval = ctk.CTkLabel(self.statistics, text = f"Amount of times hands have been removed: {self.handRemovals}", font = ("Segoe UI", 10, "bold"))
        self.statisticsHandRemoval.grid(row = 1, column = 0, pady=10, padx=10, sticky = "nsew")

        self.statisticsCorrectMaterials = ctk.CTkLabel(self.statistics, text = f"Correct materials gathered: {self.correctMaterials}%", font = ("Segoe UI", 10, "bold"))
        self.statisticsCorrectMaterials.grid(row = 2, column = 0, pady=10, padx=10, sticky = "nsew")

        self.update()

    def processVideo(self):
        processedVideoList = ["","","","",""]
        statistics = ["","","","",""]
        self.count = 0

        def sanitiseCheck(videoName: str, index: int):
            footage = cv.VideoCapture(videoName)
            videoWriter, processVideoName = camera.createVideoWriter(f"{self.dateStr}/process/{index}", self.master.cameraProperties)
            
            resultList = []

            while True:
                ret, frame = footage.read()

                if ret == True:
                    result = self.master.handDetector.detect(frame, self.count)
                    resultList.append(result)
                    self.count += 1

                elif ret == False:
                    break

            footage.release()
            resultList = self.master.handDetector.smoothHandLandmarks(resultList)
            resultList = iter(resultList) # Need an iterator for results

            footage = cv.VideoCapture(videoName)
            
            mask = SanitisationMask(self.master.cameraProperties)
            while True:
                ret, frame = footage.read()

                if ret == True:
                    result = next(resultList)
                    annotatedFrame = self.master.handDetector.paintImage(frame, result, mask)
                    videoWriter.write(annotatedFrame)

                elif ret == False:
                    break
            
            footage.release()
        
            coverage = mask.calculateMaskCoverage()
            videoWriter.release()
            processedVideoList[index] = processVideoName
            statistics[index] = coverage

        def materialCheck(videoName: str, index: int):
            footage = cv.VideoCapture(videoName)
            videoWriter, processVideoName = camera.createVideoWriter(f"{self.dateStr}/process/{index}", self.master.cameraProperties)

            while True:
                ret, frame = footage.read()

                if ret == True:
                    result = self.master.objectDetector.detect(frame, int(time.time()))
                    annotatedFrame, items, _ = self.master.objectDetector.visualiseAll(frame,result)
                    videoWriter.write(annotatedFrame)

                elif ret == False:
                    break

            if items:
                materials = True
            else:
                materials = False
                
            footage.release()
            videoWriter.release()
            processedVideoList[index] = processVideoName
            statistics[index] = materials

        def emptyCheck(videoName: str, index: int):
            footage = cv.VideoCapture(videoName)
            videoWriter, processVideoName = camera.createVideoWriter(f"{self.dateStr}/process/{index}", self.master.cameraProperties)

            while True:
                ret, frame = footage.read()

                if ret == True:
                    result = self.master.objectDetector.detect(frame, int(time.time()))
                    annotatedFrame, _, itemCount = self.master.objectDetector.visualiseAll(frame,result)
                    videoWriter.write(annotatedFrame)

                elif ret == False:
                    break

            if itemCount != 0:
                emptyStatus = False
            else:
                emptyStatus = True

            footage.release()
            videoWriter.release()
            processedVideoList[index] = processVideoName
            statistics[index] = emptyStatus

        def toolUsageCheck(videoName: str, index: int):
            footage = cv.VideoCapture(videoName)
            videoWriter, processVideoName = camera.createVideoWriter(f"{self.dateStr}/process/{index}", self.master.cameraProperties)
            handResultList = []
            objectResultList = []

            if self.usage == 0:
                tool = 'Single Channel Pipettor'
            elif self.usage == 1:
                tool = 'Pipette Controller'

            while True:
                ret, frame = footage.read()

                if ret == True:
                    resultHand = self.master.handDetector.detect(frame, self.count)
                    handResultList.append(resultHand)

                    resultObject = self.master.objectDetector.detect(frame, self.count)
                    objectResultList.append(resultObject)

                    self.count += 1

                elif ret == False:
                    break

            footage.release()
            handResultList = self.master.handDetector.smoothHandLandmarks(handResultList)
            handResultList = iter(handResultList) # Need an iterator for results

            objectResultList = iter(objectResultList) # Need an iterator for results

            footage = cv.VideoCapture(videoName)
            handsRemovedCount = 0
            prevHandCount = 0
            
            while True:
                ret, frame = footage.read()

                if ret == True:
                    resultHand = next(handResultList)
                    resultObject = next(objectResultList)
                    
                    annotatedFrame = self.master.handDetector.draw_landmarks_on_image(frame, resultHand)
                    annotatedFrame, found = self.master.objectDetector.visualiseSpecificItem(annotatedFrame, resultObject, tool)
                    
                    if len(resultHand.result.handedness) < prevHandCount or not found:
                        handsRemovedCount += 1

                    cv.putText(annotatedFrame, f"Analysing {tool} Usage",
                    (0, 25), cv.FONT_HERSHEY_DUPLEX,
                    ObjectDetectorYOLO.FONT_SIZE, (0,0,0), ObjectDetectorYOLO.FONT_THICKNESS, cv.LINE_AA)

                    cv.putText(annotatedFrame, f"Number of times hands have been removed: {handsRemovedCount}",
                    (0, 50), cv.FONT_HERSHEY_DUPLEX,
                    ObjectDetectorYOLO.FONT_SIZE/2, (0,0,0), ObjectDetectorYOLO.FONT_THICKNESS, cv.LINE_AA)

                    videoWriter.write(annotatedFrame)
                    prevHandCount = len(resultHand.result.handedness)

                elif ret == False:
                    break

            footage.release()
            videoWriter.release()
            processedVideoList[index] = processVideoName
            statistics[index] = handsRemovedCount

        sanitiseCheck(self.master.videoName[0], 0)
        print("Finished processing starting sanitisation")

        materialCheck(self.master.videoName[1], 1)
        print("Finished processing collection")

        toolUsageCheck(self.master.videoName[2], 2)
        print("Finished processing tool usage")

        emptyCheck(self.master.videoName[3], 3)
        print("Finished processing space disposal")

        sanitiseCheck(self.master.videoName[4], 4)
        print("Finished processing sanitisation")


        print("Finished all processing")

        self.master.videoName = processedVideoList

        [self.startSanitisation, self.correctMaterials, self.handRemovals, self.emptySpace, self.endSanitisation] = statistics

        self.playButton.configure(text = "Play Video")         
        self.playButton.configure(command = self.playVideo) 
        self.update()


    def playVideo(self):
        """Function to play the video on the screen
        """
        
        self.titleLabel.configure(text = "Playing video")
        self.update()

        videoWriter, videoName = camera.createVideoWriter(f"{self.dateStr}/final/final", self.master.cameraProperties)

        print(f"Now replaying footage {videoName}")

        for video in self.master.videoName:
            footage = cv.VideoCapture(video)
            if not footage.isOpened():
                 print("pain")
            else:
                print(f"Playing {video}")

                while True:
                    ret, frame = footage.read()

                    if ret:
                        videoWriter.write(frame)

                        framePIL = convertCVtoPIL(frame)
                        
                        self.cameraFrame.create_image(0, 0, anchor=tk.NW, image=framePIL)
                        self.cameraFrame.image = framePIL
                        self.update()

                    if ret == False or cv.waitKey(1) == ord('q'):
                        break
                
            footage.release()
        
        
        videoWriter.release()

        self.playButton.configure(text = "Return to menu")         
        self.playButton.configure(command = self.exit) 
    
    def exit(self):
        self.master.displayMenu()
        self.destroy


if __name__ == "__main__":
    displayApp = app()

    displayApp.mainloop()