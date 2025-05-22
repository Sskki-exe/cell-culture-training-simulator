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
# from visualizer3d import visualizer3dSCPVideo, visualizer3dAIDVideo, transMatrix, scenePyRender
import numpy as np
import pyrender
import trimesh
from reportMaker import makeReport
from pandaVisual import *

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

def serial2Dict(serialString: str):
    """Function to convert from the string outputted from the Arduino into a dictionary for CSV saving.

    Args:
        serialString (str): String from Arduino

    Returns:
        dict: Dictionary explaining what string was
    """
    SCP, AID = serialString.split("+")
    label, data = SCP.split(":")
    rollSCP, pitchSCP, buttonSCP = data.split("/")
    label, data = AID.split(":")
    rollAID, pitchAID, buttonAID = data.split("/")
    row = {
        "SCPRollF":rollSCP,
        "SCPPitchF":pitchSCP,
        "SCPButton":buttonSCP,
        "AIDRollF":rollAID,
        "AIDPitchF":pitchAID,
        "AIDButton":buttonAID
    }
    return row

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
            self.destroy()

        # Change camera properties to be as best as they can be for more accurate detection
        self.cameraProperties = camera.getCameraProperties(self.cap)

        # Loading Screen 2
        self.display = loadingFrame(self, loadingImageStr="loadingGraphics/2.JPG")
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()

        # Model Initalisation
        self.handDetector = handLandmarker(cameraProperties=self.cameraProperties)
        self.objectDetector = ObjectDetectorYOLO()
        self.function = 0
        self.handLength = 20
        self.digitalTwin = PandaRenderer(self.cameraProperties)

        # Results interpreted from most recent recording
        self.videoName = [] 
        self.toolData = ""
        self.toolSampleRange = None

        # Show loading menu
        # self.displayFeedback('Single Channel Pipettor',"160525 150457")
        self.display = menuFrame(self)
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()

    def displayMenu(self):
        """Function to display menu
        """
        self.handDetector = handLandmarker(cameraProperties=self.cameraProperties, handLength=self.handLength)
        
        self.videoName = []         
        self.toolData = ""
        self.toolSampleRange = None
        self.display = menuFrame(self)
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()

    def displayPractiseSanitation(self):
        """Function to record screen for Single Channel Pipette Use
        """
        self.handDetector = handLandmarker(cameraProperties=self.cameraProperties, handLength=self.handLength)
        self.display = practiseHandScreen(self)
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()

    def displaySingleChannelPractise(self):
        """Function to record screen for Single Channel Pipette Use
        """
        self.display = practiseToolScreen(self, 0)
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()

    def displayPipetteAidPractise(self):
        """Function to record screen for Pipette Aid Use
        """
        self.display = practiseToolScreen(self, 1)
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()
    
    def displaySingleChannelTest(self):
        """Function to record screen for Single Channel Pipette Use
        """
        self.tool = 0
        self.display = testScreen(self, 0)
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()

    def displayPipetteAidTest(self):
        """Function to record screen for Pipette Aid Use
        """
        self.tool = 1
        self.display = testScreen(self, 1)
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()
    
    def displaySettings(self):
        """Function to playback video with annotations

        Args:
            usage (str): One word description of what the recording was about
        """
        self.display = settingsScreen(self)
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()

    def displayFeedback(self, usage: str, dateStr):
        """Function to playback video with feedback

        Args:
            usage (str): One word description of what the recording was about
        """
        self.display = feedbackScreen(self, usage, dateStr)
        self.display.grid(row=0, column = 0, pady=10, padx=10, sticky = "nsew")
        self.update()
    
    def setHandLength(self, handLength):
        """Function to recalibrate hand length
        """
        self.handLength = handLength
        self.displayPractiseSanitation()
    
    def close(self):
        """Function to destroy app from menu screen
        """
        self.cap.release()
        self.destroy()

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

class menuFrame(ctk.CTkFrame):
    """Frame used as the application menu
    """
    def __init__(self, master, fg_color = None):
        super().__init__(master, width = 2000, height = 2000)

        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure(0, weight = 1)
        self.grid_rowconfigure(1, weight = 0, uniform="cell") # Practise Sanitation
        self.grid_rowconfigure(2, weight = 0, uniform="cell") # Practise Single Channel Pipette
        self.grid_rowconfigure(3, weight = 0, uniform="cell") # Practise Pipette Aid
        self.grid_rowconfigure(4, weight = 0, uniform="cell") # Assess Single Channel Pipette
        self.grid_rowconfigure(5, weight = 0, uniform="cell") # Assess Pipette Aid
        self.grid_rowconfigure(6, weight = 0, uniform="cell") # Calibrate
        self.grid_rowconfigure(7, weight = 0, uniform="cell") # Close App

        self.titleLabel = ctk.CTkLabel(self, text = "Cell Culture Training", font = ("Segoe UI", 100, "bold"))
        self.titleLabel.grid(row = 0, column = 0, pady=10, padx=10, sticky = "new")

        self.sanitationButton = ctk.CTkButton(self, text = "Practise Sanitation", fg_color="#EC9006", hover_color="#E27602", command = master.displayPractiseSanitation)
        self.sanitationButton.grid(row = 1, column = 0, pady=10, padx=10, sticky = "nsew")
        
        self.singleChannelPippettePractiseButton = ctk.CTkButton(self, text = "Practise Single Channel Pippetting", fg_color="#EC9006", hover_color="#E27602", command = master.displaySingleChannelPractise)
        self.singleChannelPippettePractiseButton.grid(row = 2, column = 0, pady=10, padx=10, sticky = "nsew")

        self.pippetteAidPractiseButton = ctk.CTkButton(self, text = "Practise Pipette Aid", fg_color="#EC9006", hover_color="#E27602", command = master.displayPipetteAidPractise)
        self.pippetteAidPractiseButton.grid(row = 3, column = 0, pady=10, padx=10, sticky = "nsew")

        self.singleChannelPippetteTestButton = ctk.CTkButton(self, text = "Assess Single Channel Pippetting", command = master.displaySingleChannelTest)
        self.singleChannelPippetteTestButton.grid(row = 4, column = 0, pady=10, padx=10, sticky = "nsew")

        self.pippetteAidTestButton = ctk.CTkButton(self, text = "Assess Pipette Aid", command = master.displayPipetteAidTest)
        self.pippetteAidTestButton.grid(row = 5, column = 0, pady=10, padx=10, sticky = "nsew")

        self.settingsButton = ctk.CTkButton(self, text = "Settings", fg_color="#006400", hover_color="#003C00", command = master.displaySettings)
        self.settingsButton.grid(row = 6, column = 0, pady=10, padx=10, sticky = "nsew")
         
        self.escapeButton = ctk.CTkButton(self, text = "Exit", command = master.close, fg_color="#8B0000", hover_color="#610000")
        self.escapeButton.grid(row = 7, column = 0, pady=10, padx=10, sticky = "nsew")

class practiseHandScreen(ctk.CTkFrame):
    """Frame used to practise sanitation
    """
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.mask = SanitisationMask(self.master.cameraProperties)
        self.result = None

        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 1) 
        self.grid_rowconfigure(2, weight = 0)
        self.grid_rowconfigure(3, weight = 0) 
        self.grid_rowconfigure(4, weight = 0) 

        self.titleLabel = ctk.CTkLabel(self, text = f"Practise Sanitation", font = ("Segoe UI", 80, "bold"))
        self.titleLabel.grid(row = 0, column = 0, pady=10, padx=10, sticky = "nsew")

        self.cameraFrame = tk.Canvas(self, width = self.master.cameraProperties[0], height = self.master.cameraProperties[1], highlightthickness=1)
        self.cameraFrame.grid(row = 1, column = 0, pady = 10, padx = 10)

        self.resetButton = ctk.CTkButton(self, text = "Reset Sanitation", command = self.mask.resetMask, fg_color="#EC9006", hover_color="#E27602")
        self.resetButton.grid(row = 2, column = 0, pady=10, padx=10, sticky = "nsew")

        self.calibrateButton = ctk.CTkButton(self, text = "Calibrate Hands", command = self.setHandLength, fg_color="#EC9006", hover_color="#E27602")
        self.calibrateButton.grid(row = 3, column = 0, pady=10, padx=10, sticky = "nsew")

        self.escapeButton = ctk.CTkButton(self, text = "Return to Main Menu", command = self.exitFrame, fg_color="#8B0000", hover_color="#610000")
        self.escapeButton.grid(row = 4, column = 0, pady=10, padx=10, sticky = "nsew")

        self.updateCameraFrame()
    
    def exitFrame(self):
        self.master.displayMenu()
        self.destroy()
    
    def updateCameraFrame(self):
        ret, frame = self.master.cap.read()
        if ret:
            self.result = self.master.handDetector.detect(frame, int(time.monotonic()*1000))
            annotatedFrame = self.master.handDetector.paintImage(frame, self.result, self.mask)
            framePIL = convertCVtoPIL(annotatedFrame)
            
            self.cameraFrame.create_image(0, 0, anchor=tk.NW, image=framePIL)
            self.cameraFrame.image = framePIL
            self.update()
            
        self.after(33, self.updateCameraFrame)  # ~30 FPS
    
    def setHandLength(self):
        detection_result = self.result.result.hand_landmarks[0]
        middleKnuckle = np.array(self.master.handDetector.convertNormalToImageCoord(detection_result[9].x,detection_result[9].y)) # Gets the coordinate of the middle knuckle
        wrist = np.array(self.master.handDetector.convertNormalToImageCoord(detection_result[0].x,detection_result[0].y)) # Gets the coordinate of the wrist
        pixelDist = np.linalg.norm(middleKnuckle-wrist) # Figure out the amount of pixels used
        self.master.setHandLength(pixelDist)
        self.destroy()

class practiseToolScreen(ctk.CTkFrame):
    """Frame used to practise tool
    """
    def __init__(self, master, usage):
        super().__init__(master)
        self.master = master
        self.usage = usage
        
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

        self.titleLabel = ctk.CTkLabel(self, text = f"Practise {self.usage}", font = ("Segoe UI", 80, "bold"))
        self.titleLabel.grid(row = 0, column = 0, pady=10, padx=10, sticky = "nsew")

        self.cameraFrame = tk.Canvas(self, width = self.master.cameraProperties[0], height = self.master.cameraProperties[1], highlightthickness=1)
        self.cameraFrame.grid(row = 1, column = 0, pady = 10, padx = 10)

        self.escapeButton = ctk.CTkButton(self, text = "Return to Main Menu", command = self.close, fg_color="#8B0000", hover_color="#610000")
        self.escapeButton.grid(row = 2, column = 0, pady=10, padx=10, sticky = "nsew")
        
        self.cameraFrame.update()

        self.scene = self.master.digitalTwin
        self.string = "Pipette_1:0/0/0+Aid_1:0/0/00" #Sample
        
        self.updateCameraFrame()

    def updateCameraFrame(self):
        startTime = time.time()
        string = read_hub_serial()
        try:
            data = serial2Dict(string)
            self.string = string
        except:
            data = serial2Dict(self.string)
        """
        data = {
        "SCPRollF":rollSCP,
        "SCPPitchF":pitchSCP,
        "SCPButton":buttonSCP,
        "AIDRollF":rollAID,
        "AIDPitchF":pitchAID,
        "AIDButton":buttonAID}
        """
        if self.no == 0:
            roll = float(data['SCPRollF'])
            pitch = float(data['SCPPitchF'])
            button = int(data['SCPButton'])

            if button == 0:  # button down
                mesh = "3dassets/pipette_up.obj"
                buttonTEXT = "False"
            elif button == 1:  # button down
                mesh = "3dassets/pipette_down.obj"
                buttonTEXT = "True"

        elif self.no == 1:
            roll = float(data['AIDRollF'])
            pitch = float(data['AIDPitchF'])
            button = int(data['AIDButton'])
            if button == 0:  # IDLE
                mesh = "3dassets/pipette_up.obj"
                buttonTEXT = "Idle"
            elif button == 1:  # SUCK
                mesh = "3dassets/pipette_down.obj"
                buttonTEXT = "Sucking"
            elif button == 10: # RELEASE
                mesh = "3dassets/pipette_up.obj"
                buttonTEXT = "Releasing"
            elif button == 11: # IDLE
                mesh = "3dassets/pipette_down.obj"
                buttonTEXT = "Idle"

        T = transMatrix(np.deg2rad(roll),np.deg2rad(pitch))
        img_bgr = self.scene.render_mesh(mesh, T) # Scene Renderer

        # Add text to the frame
        cv.putText(img_bgr, f"{self.usage}", (0, 25), cv.FONT_HERSHEY_DUPLEX,
                   0.5, (0, 0, 0), 1, cv.LINE_AA)

        cv.putText(img_bgr, f"Roll: {round(roll, 2)}, Pitch: {round(pitch, 2)}, Button Pressed: {buttonTEXT}",
                   (0, 50), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        imgPIL = convertCVtoPIL(img_bgr)

        self.cameraFrame.create_image(0, 0, anchor=tk.NW, image=imgPIL)
        self.cameraFrame.image = imgPIL
        # print(time.time() - startTime)
        self.update()            
        self.after(33, self.updateCameraFrame)  # ~30 FPS
    
    def close(self):
        self.master.displayMenu()
        self.destroy()

class settingsScreen(ctk.CTkFrame):
    """Frame used to change settings in the app
    """
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 0) 
        self.grid_rowconfigure(2, weight = 0)

        self.titleLabel = ctk.CTkLabel(self, text = f"Settings", font = ("Segoe UI", 80, "bold"))
        self.titleLabel.grid(row = 0, column = 0, pady=10, padx=10, sticky = "nsew")

        self.toggleAppearanceButton = ctk.CTkButton(self, text = "Toggle Appearance", command=self.toggleAppearance)
        self.toggleAppearanceButton.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = "nsew")

        self.escapeButton = ctk.CTkButton(self, text = "Return to Main Menu", command = self.master.displayMenu, fg_color="#8B0000", hover_color="#610000")
        self.escapeButton.grid(row = 3, column = 0, pady=10, padx=10, sticky = "nsew")
    
    def toggleAppearance(self):
        currentMode = ctk.get_appearance_mode()

        if currentMode == 'Dark':
            ctk.set_appearance_mode("Light")
        
        elif currentMode == 'Light':
            ctk.set_appearance_mode("Dark")

class testScreen(ctk.CTkFrame):
    """Frame used to test people's skills
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
        
        self.titleLabel = ctk.CTkLabel(self, text = f"Recording", font = ("Segoe UI", 80, "bold"))
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


        file = open(f"video/{dateStr}/raw/data.csv", mode='w', newline='')
        dataWriter = csv.DictWriter(file, fieldnames=["SCPRollF","SCPPitchF","SCPButton","AIDRollF","AIDPitchF","AIDButton"])
        
        def toolSampler(csvWriter):
            """Function that will sample the position of an object and write to a dictionary

            Args:
                csvWriter (csv.DictWriter): CSV file that the data will be written into
            """
            string = read_hub_serial()
            strDict = serial2Dict(string)
            csvWriter.writerow(strDict)

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
        self.titleLabel.configure(text = "Recording Sanitation")
        self.update()

        start = time.time()

        count = 0
        sampleCount = [0,0]
        while self.recordCamera == True:
            ret, frame = self.master.cap.read()
            if ret:
                videoWriter.write(frame)
                toolSampler(dataWriter)
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
        sampleCount[0] = sampleCount[0] + count
        
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
        self.titleLabel.configure(text = "Recording Material Collection")
        self.update()
        
        start = time.time()

        count = 0
        while self.recordCamera == True:
            ret, frame = self.master.cap.read()
            if ret:
                videoWriter.write(frame)
                toolSampler(dataWriter)
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
        sampleCount[0] = sampleCount[0] + count

        # Step 3: Use Tool
        videoWriter, videoName = camera.createVideoWriter(function=f"{dateStr}/raw/tool", cameraProperties=self.master.cameraProperties)
        self.recordButton.configure(command = self.endVideo)
        self.recordButton.configure(text = f"Start cleaning space")
        self.update()

        if not videoWriter.isOpened():
            print("AHHHHHHHHHHHH")
            exit()

        print("Start recording tool usage")
        self.recordCamera = True
        self.titleLabel.configure(text = "Recording Tool Usage")
        self.update()

        start = time.time()

        count = 0
        while self.recordCamera == True:
            ret, frame = self.master.cap.read()
            if ret:
                videoWriter.write(frame)
                toolSampler(dataWriter)
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
        sampleCount[1] = sampleCount[0] + count

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
        self.titleLabel.configure(text = "Recording space cleaning")
        self.update()

        start = time.time()

        count = 0
        while self.recordCamera == True:
            ret, frame = self.master.cap.read()
            if ret:
                videoWriter.write(frame)
                toolSampler(dataWriter)
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
        self.titleLabel.configure(text = "Recording Sanitation")
        self.update()

        start = time.time()

        count = 0
        while self.recordCamera == True:
            ret, frame = self.master.cap.read()
            if ret:
                videoWriter.write(frame)
                toolSampler(dataWriter)
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
        self.master.toolData = f"video/{dateStr}/raw/data.csv"
        file.close()
        sampleCount[0] = sampleCount[0]-1
        self.master.toolSampleRange = sampleCount
        self.master.displayFeedback(self.no, dateStr)
        self.destroy()

    def endVideo(self):
        """Function to end the recording of the video, as triggered by a button.
        """
        self.recordCamera = False
        # self.cameraFrame = ctk.CTkLabel(self, fg_color="transparent", image = self.placeholder, text = "")
        # self.cameraFrame.grid(row = 1, column = 0, pady = 10, padx = 10, sticky = "nsew")
        self.update()

class feedbackScreen(ctk.CTkFrame):
    """Frame to play video after processing and analysis
    """
    def __init__(self, master, usage: int, dateStr: str):
        super().__init__(master)
        self.master = master
        self.usage = usage
        self.dateStr = dateStr

        # When to analyze tools
        self.toolSampleRange = self.master.toolSampleRange

        # Graphics
        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure(0, weight = 0)
        self.grid_rowconfigure(1, weight = 1) 
        self.grid_rowconfigure(2, weight = 1) 
    
        ### PUT PLACEHOLDER IMAGE WHILST CAMERA IS OFF
        self.placeholder = ctk.CTkImage(light_image=Image.open("loadingGraphics/wait.jpg"),
                            dark_image=Image.open("loadingGraphics/wait.jpg"),
                            size=(1280, 720))
        
        self.titleLabel = ctk.CTkLabel(self, text = f"Reviewing", font = ("Segoe UI", 80, "bold"))
        self.titleLabel.grid(row = 0, column = 0, pady=10, padx=10, sticky = "nsew")

        self.cameraFrame = tk.Canvas(self, width = self.master.cameraProperties[0]*3, height = self.master.cameraProperties[1], highlightthickness=1)
        self.cameraFrame.grid(row = 1, column = 0, pady = 10, padx = 10)

        tempFrame = Image.open("loadingGraphics/wait.jpg")
        tempFrame = ImageTk.PhotoImage(image=tempFrame)

        self.displayImage = self.cameraFrame.create_image(0, 0, anchor=tk.NW, image=tempFrame)
        self.cameraFrame.image = tempFrame

        self.playButton = ctk.CTkButton(self, text = "Process video", command = self.processVideo)
        self.playButton.grid(row = 3, column = 0, pady=10, padx=10, sticky = "nsew")
       
        # Statistics frame
        self.statistics = ctk.CTkFrame(self)
        self.statistics.grid(row = 2, column = 0, pady=10, padx=10)
        self.statistics.pack_propagate(False)
        
        self.details = ctk.CTkTextbox(self.statistics, width=self.master.cameraProperties[0]*2-50, height = cameraProperties[1]-100)
        self.details.grid(row=0,column=0, pady = 10, padx = 10, sticky = "nsew")

        self.statistics.grid_columnconfigure(0, weight = 1)
        self.statistics.grid_rowconfigure(0, weight = 1)

        self.update()

    def processVideo(self):
        """Function to process all the videos made.
        """
        processedVideoList = ["","","","",""]
        textFile = open(f"video/{self.dateStr}/final/note.txt", "a")

        def sanitiseCheck(videoName: str, index: int):
            footage = cv.VideoCapture(videoName)
            videoWriter, processVideoName = camera.createVideoWriter(f"{self.dateStr}/process/{index}", self.master.cameraProperties)
            
            resultList = []

            while True: # Go through video and get initial positions
                ret, frame = footage.read()

                if ret == True:
                    result = self.master.handDetector.detect(frame, int(time.monotonic()*1000))
                    resultList.append(result)

                elif ret == False:
                    break

            footage.release()
            resultList = self.master.handDetector.smoothHandLandmarks(resultList) # Try to smooth results to prevent FN
            resultList = iter(resultList) # Need an iterator for results

            footage = cv.VideoCapture(videoName)
            
            mask = SanitisationMask(self.master.cameraProperties)
            while True: # Annotate video with sanitation coverage.
                ret, frame = footage.read()

                if ret == True:
                    result = next(resultList)
                    annotatedFrame = self.master.handDetector.paintImage(frame, result, mask)
                    videoWriter.write(annotatedFrame) # Save annotated frame into video

                elif ret == False:
                    break
            
            footage.release()
        
            coverage = mask.calculateMaskCoverage() # Get percentage
            mask.saveMaskImage(filename=f"video/{self.dateStr}/process/{index}.png")
            videoWriter.release()
            processedVideoList[index] = processVideoName # Store video name

            if index == 0:
                textFile.write(f"Your initial sanitiation had a total of {coverage}% coverage.")
                textFile.write("\n")
            else:
                textFile.write(f"Your final sanitiation had a total of {coverage}% coverage.")
                textFile.write("\n")

        def materialCheck(videoName: str, index: int):
            footage = cv.VideoCapture(videoName)
            videoWriter, processVideoName = camera.createVideoWriter(f"{self.dateStr}/process/{index}", self.master.cameraProperties)
            # itemsList = ["","","","",""] # I think there are issues with sometimes detection not occuring, so I might have a buffer of items just incase

            while True: # Annotate video with items detected
                ret, frame = footage.read()

                if ret == True:
                    result = self.master.objectDetector.detect(frame, int(time.time()))
                    annotatedFrame, items, _ = self.master.objectDetector.visualiseAll(frame,result) # Mark all items in Frame.
                    videoWriter.write(annotatedFrame) # Save annotated frame into video

                    # For buffering
                    # itemsList[0:4] = itemsList[1:]
                    # itemsList[4] = items

                elif ret == False:
                    break

            self.master.objectDetector.saveObjectLocations(annotatedFrame, result, f"video/{self.dateStr}")
            
            textFile.write(f"These are the items you collected:")
            textFile.write(', '.join(f"{key}: {value}" for key, value in items.items()))
            textFile.write(".\t")     

            # Hardcoded beyond belief
            neededItems = dict()
            neededItems["Single Channel Pipettor"] = 1
            neededItems['Cell Culture Plate - 24'] = 1

            missingItems, missingBool = self.master.objectDetector.checkCorrectItems(items, neededItems) # Check
       
            if missingBool:
                textFile.write(f"You got all the materials you needed!")
                textFile.write("\n")
            else:
                textFile.write(f"Unfortunately, you didn't get all the items you needed. You are still missing: ")
                textFile.write(', '.join(f"{key}: {value}" for key, value in missingItems.items()))
                textFile.write(".\n")
                
            footage.release()
            videoWriter.release()
            processedVideoList[index] = processVideoName

        def emptyCheck(videoName: str, index: int):
            footage = cv.VideoCapture(videoName)
            videoWriter, processVideoName = camera.createVideoWriter(f"{self.dateStr}/process/{index}", self.master.cameraProperties)
            
            while True:
                ret, frame = footage.read()

                if ret == True:
                    result = self.master.objectDetector.detect(frame, int(time.time()))
                    annotatedFrame, _, itemCount = self.master.objectDetector.visualiseAll(frame,result)
                    videoWriter.write(annotatedFrame) # Save annotated frame into video

                elif ret == False:
                    break

            if itemCount == 0:
                textFile.write(f"Great job! You fully emptied the space.")
                textFile.write("\n")
            else:
                textFile.write(f"Unfortunately, you didn't take out all the items; you left {itemCount} items behind. Remember to remove all items before sanitising at the end.")
                textFile.write("\n")

            footage.release()
            videoWriter.release()
            processedVideoList[index] = processVideoName

        def toolUsageCheck(videoName: str, index: int):
            footage = cv.VideoCapture(videoName)
            videoWriter, processVideoName = camera.createVideoWriter(f"{self.dateStr}/process/{index}", self.master.cameraProperties)
            handResultList = [] # During this section, need to measure both hands and objects
            objectResultList = [] # During this section, need to measure both hands and objects

            if self.usage == 0:
                tool = 'Single Channel Pipettor'
            elif self.usage == 1:
                tool = 'Pipette Controller'

            while True: # Do first pass through of detections
                ret, frame = footage.read()

                if ret == True:
                    resultHand = self.master.handDetector.detect(frame, int(time.monotonic()*1000))
                    handResultList.append(resultHand)

                    resultObject = self.master.objectDetector.detect(frame, int(time.monotonic()*1000))
                    objectResultList.append(resultObject)

                elif ret == False:
                    break

            footage.release()
            handResultList = self.master.handDetector.smoothHandLandmarks(handResultList) # Smooth the hands
            handResultList = iter(handResultList) # Need an iterator for results

            objectResultList = iter(objectResultList) # Need an iterator for results

            footage = cv.VideoCapture(videoName)
            handsRemovalCount = 0
            prevHandCount = 0
            prevResultHand = None
            maxSpeed = 0
            
            while True: # Do second pass to do annotations and do analysis.
                ret, frame = footage.read()

                if ret == True:
                    resultHand = next(handResultList)
                    resultObject = next(objectResultList)

                    annotatedFrame = self.master.handDetector.draw_landmarks_on_image(frame, resultHand) # Do annotations
                    annotatedFrame, _ = self.master.objectDetector.visualiseSpecificItem(annotatedFrame, resultObject, tool)

                    curHandCount = len(resultHand.result.handedness) # Count the number of hands
                    if prevHandCount > curHandCount:
                        handsRemovalCount += 1
                    
                    prevHandCount = curHandCount

                    if prevResultHand: # Get the speed of the hands
                        speed = self.master.handLandmarker.getHandSpeed(prevResultHand.result, resultHand.result)
                        if maxSpeed < speed:
                            maxSpeed = speed
                    
                    videoWriter.write(annotatedFrame) # Save annotated frame into video

                elif ret == False:
                    break

            badSpeed = 30 # Speed which we deem to be too fast; in cm/s
            if maxSpeed < badSpeed:
                textFile.write(f"Your hands had a max speed of {maxSpeed} cm/s, which is below the recommended hand speed of {badSpeed} cm/s. Great job!\t")
            
            else:
                textFile.write(f"Your hands had a max speed of {maxSpeed} cm/s, which reached beyond the recommended hand speed of {badSpeed} cm/s.\t")

            textFile.write(f"You removed your hands a total of {handsRemovalCount} times.\t")

            footage.release()
            videoWriter.release() 
            processedVideoList[index] = processVideoName

            # Finish analysing video, now analysing tool
            if self.usage == 0:
                with open(self.master.toolData, mode="r", newline='') as file:
                    allData = list(csv.DictReader(file, fieldnames = ['roll','pitch','button','a','b','c']))  # Read all rows into a list
                    toolData = allData[self.toolSampleRange[0]:self.toolSampleRange[1]]  # Adjust for 0-based indexing
            elif self.usage == 1:
                with open(self.master.toolData, mode="r", newline='') as file:
                    allData = list(csv.DictReader(file, fieldnames = ['a','b','c','roll','pitch','button']))  # Read all rows into a list
                    toolData = allData[self.toolSampleRange[0]:self.toolSampleRange[1]]  # Adjust for 0-based indexing

            
            # Note: 210 - 330 is okay curl
            badUseCount = 0
            badAngle = False # Bool that resets after every falling edge

            if self.usage == 0:
                for sample in toolData:
                    if sample["button"] == 1: # Check if button is pressed aka liquid is being picked up/dropped
                        rollGood = 210 <= float(sample["roll"]) <= 260  or 280 <= float(sample["roll"]) <= 330 # check if either pitch or roll is okay
                        pitchGood = 210 <= float(sample["pitch"]) <= 260 or 280 <= float(sample["pitch"]) <= 330 # check if either pitch or roll is okay
                        if not (rollGood or pitchGood): # if neither is good:
                            if not badAngle: # no point counting it during a long press
                                badAngle = True
                                badUseCount += 1 # Increase baduse count by 1

                    else:
                        badAngle = False # Reset after falling edge

            elif self.usage == 1:
                for sample in toolData:
                    if sample["button"] == '01' or sample["button"] == '10': # Check if button is pressed aka liquid is being picked up/dropped
                        rollGood = 210 <= float(sample["roll"]) <= 260  or 280 <= float(sample["roll"]) <= 330 # check if either pitch or roll is okay
                        pitchGood = 210 <= float(sample["pitch"]) <= 260 or 280 <= float(sample["pitch"]) <= 330 # check if either pitch or roll is okay
                        if not (rollGood or pitchGood): # if neither is good:
                            if not badAngle: # no point counting it during a long press
                                badAngle = True
                                badUseCount += 1 # Increase baduse count by 1

                    else:
                        badAngle = False # Reset after falling edge


            textFile.write(f"When using the {tool}, you held it at a bad angle {badUseCount} times. Make sure to hold it at a slight angle instead of vertically upwards to prevent contamination.")
            textFile.write("\n")

        ##########################################################################################
        # Order of video check.
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

        SCPVideoName = visualizer3dSCPVideoPanda(self.master.toolData, self.master.cameraProperties, self.master.digitalTwin, self.dateStr)
        AIDVideoName = visualizer3dAIDVideoPanda(self.master.toolData, self.master.cameraProperties, self.master.digitalTwin, self.dateStr)
        print("Finished processing tool use")

        print("Finished all processing")
        makeReport(f"video/{self.dateStr}")
        print("Report generated")

        self.master.toolData = [SCPVideoName,AIDVideoName]
        self.master.videoName = processedVideoList

        self.playButton.configure(text = "Play Video")         
        self.playButton.configure(command = self.playVideo) 
        self.update()

    def playVideo(self):
        """Function to play the video on the screen
        """
        
        self.titleLabel.configure(text = "Playing video")
        self.update()

        cameraProperties = self.master.cameraProperties.copy()
        cameraProperties[0] = 3 * cameraProperties[0]

        videoWriter, videoName = camera.createVideoWriter(f"{self.dateStr}/final/final", cameraProperties)

        print(f"Now replaying footage {videoName}")
        SCPFootage = cv.VideoCapture(self.master.toolData[0])
        if not SCPFootage.isOpened():
            print("pain")

        AIDFootage = cv.VideoCapture(self.master.toolData[1])
        if not AIDFootage.isOpened():
            print("pain")

        notes = open(f"video/{self.dateStr}/final/note.txt", "r")
        noteAll = ""

        for video in self.master.videoName:
            camFootage = cv.VideoCapture(video)
            if not camFootage.isOpened():
                 print("pain")
            else:
                print(f"Playing {video}")

                while True:
                    retCam, frameCam = camFootage.read()
                    retSCP, frameSCP = SCPFootage.read()
                    retAID, frameAID = AIDFootage.read()

                    if retCam and retSCP and retAID:
                        frame = np.hstack((frameSCP,frameCam,frameAID))
                        videoWriter.write(frame)

                        framePIL = convertCVtoPIL(frame)
                        
                        self.cameraFrame.create_image(0, 0, anchor=tk.NW, image=framePIL)
                        self.cameraFrame.image = framePIL

                        self.update()

                    if retCam == False or cv.waitKey(1) == ord('q'):
                        break    
            camFootage.release()
            note = notes.readline()
            splitNote = note.replace('\t','\n\n')
            noteAll = noteAll +'\n' + splitNote
            self.details.delete("0.0","end")
            self.details.insert("0.0", text = noteAll.strip())
            self.update()
        
        SCPFootage.release()
        AIDFootage.release()
        videoWriter.release()
        videoWriter.release()

        self.playButton.configure(text = "Return to menu")         
        self.playButton.configure(command = self.exit) 
    
    def exit(self):
        self.master.displayMenu()
        self.destroy

if __name__ == "__main__":
    displayApp = app()
    cameraProperties = displayApp.cameraProperties

    displayApp.mainloop()