This is the GITHUB I will be using temporarily to test creation of a GUI which is able to run and show the camera output from our cameras. After getting feedback from the rest of the team, I will move the contents of this repo into the main repo. I know I could be making branches, but seebs./

THIS BRANCH IS FOR PROCESSING VIDEO, NOT FOR REAL TIME 

### Installing Dependencies
These files will have to use a lot of dependencies. This is a very quick tutorial (https://www.geeksforgeeks.org/create-virtual-environment-using-venv-python/) on how to create the venv. Make sure you use a version of Python between version 3.9 - 3.12
Follow these instructions up to step 4, making sure you activate the venv. I would advise you calling your venv 'cameraVENV' (exclude the apostrophes!!!!) to ensure that .gitignore actually ignores it because I was too lazy to be smart and remove specific folders lol. 
Then, run this command in your terminal window:

pip install -r requirements.txt

You should now have all your dependencies needed to run the python file.
To ensure that you are actually using this python interpreter, in VSCode, click: **Ctrl + Shift + P** and select Python: Select Interpreter. Then, pick the new VENV you've just made! Now, when you run the python file, it will actually use the interpreter.

NOTE: The 

### Files
#### [[main.py]]
The main file that will (hopefully) sucessfully run everything. It uses CTk to create the separate program, although it is kinda rough to use when working with for the first time. Documentation is here: https://customtkinter.tomschimansky.com/documentation/

#### [[handtracking.py]]
The file that holds all of the objects and classes involving MediaPipe handtracking software.

#### [[hand_landmarker.task]]
This is the model I'm using for the handtracking. It was developed by Google and can be found here (https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#get_started.).
