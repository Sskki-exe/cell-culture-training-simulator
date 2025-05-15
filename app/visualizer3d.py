import pandas as pd
import cv2 as cv
import numpy as np
import pyrender
import random
import trimesh
from camera import createVideoWriter, getCameraProperties
from datetime import datetime
import time
import open3d as o3d
import torch
from serial_read_sample import read_hub_serial

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
################################ Transformation ############################################ 
def transMatrix(roll,pitch):
    """Function to calculate transformation matrix based off roll and pitch

    Args:
        roll (np.radians): Roll of object in radians
        pitch (np.radians): Pitch of object in radians

    Returns:
        4x4 np.array: Transformation matrix based off roll and pitch
    """
    # Compute rotation matrices for roll and pitch
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])  # Rotation using roll

    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])  # Rotation using pitch

    # Combine roll and pitch rotations
    R = Rx @ Ry  
    T = np.eye(4)
    T[:3, :3] = R

    return T

################################ CPU Video ############################################ 
def scenePyRender(scene, mesh, T, cameraProperties):
    """Function to render a mesh on a scene using a 4x4 matrix

    Args:
        scene (pyrender.Scene): Scene for showing object
        mesh (trimesh): Object mesh
        T : 4x4 Transformation Matrix

    Returns:
        img_bgr, mesh_node: OpenCV frame, mesh_node
    """
    # Apply the transformation to the mesh
    mesh.apply_transform(T)

    # Create a Pyrender mesh from the Trimesh object
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)

    # Create a Pyrender node for the mesh and add it to the scene
    mesh_node = scene.add(pyrender_mesh)

    # Set up the camera (Position it in front of the mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=cameraProperties[0] / cameraProperties[1])

    # Move the camera further away from the object on the Z-axis
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = [0, 0, 30]  # Move the camera even further on the Z-axis (increase the value)
    scene.add(camera, pose=camera_pose)

    # Set up a light source in the scene
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Set up offscreen rendering with Pyrender
    r = pyrender.OffscreenRenderer(int(cameraProperties[0]), int(cameraProperties[1]))
    color, depth = r.render(scene)

    # Convert the color image to OpenCV BGR format
    img_bgr = cv.cvtColor(color, cv.COLOR_RGB2BGR)

    return img_bgr, mesh_node

def visualizer3dSCPVideoCPU(filename: str, cameraProperties: list, dateStr: str = "", test: bool = False):
    # Read the CSV file with transformation data
    dataCSV = pd.read_csv(filename, header=None)
    dataCSV.columns = ['roll', 'pitch', 'button', 'a', 'b', 'c']
    
    # Set up video writer based on test flag
    if test:
        videoWriter, videoName = createVideoWriter("vistest/model", cameraProperties)
    else:
        videoWriter, videoName = createVideoWriter(f"{dateStr}/process/modelSCP", cameraProperties)

    # Pyrender setup (create a Pyrender scene)
    scene = pyrender.Scene()

    for rowIndex, data in dataCSV.iterrows():
        roll = np.radians(data['roll'])
        pitch = np.radians(data['pitch'])
        button = data['button']

        T = transMatrix(roll,pitch) # Calculate transformation matrix

        # Load the appropriate mesh based on the button state
        if button == 0:  # button down
            mesh = trimesh.load_mesh("3dassets/pipette_up.obj")
        elif button == 1:  # button down
            mesh = trimesh.load_mesh("3dassets/pipette_down.obj")

        img_bgr, mesh_node = scenePyRender(scene,mesh,T, cameraProperties) # Scene Renderer

        # Add text to the frame
        cv.putText(img_bgr, f"Single Channel Pipette", (0, 25), cv.FONT_HERSHEY_DUPLEX,
                   0.5, (0, 0, 0), 1, cv.LINE_AA)

        cv.putText(img_bgr, f"Roll: {round(data['roll'], 2)}, Pitch: {round(data['pitch'], 2)}, Button Pressed: {bool(data['button'])}",
                   (0, 50), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        # Write the frame to the video file
        videoWriter.write(img_bgr)

        # Remove the mesh node from the scene after rendering the frame
        scene.remove_node(mesh_node)

    # Release the video writer after all frames are written
    videoWriter.release()

    return videoName

def visualizer3dAIDVideoCPU(filename: str, cameraProperties: list, dateStr: str = "", test: bool = False):
    # Read the CSV file with transformation data
    dataCSV = pd.read_csv(filename, header=None)
    dataCSV.columns = ['a', 'b', 'c','roll', 'pitch', 'button']
    
    # Set up video writer based on test flag
    if test:
        videoWriter, videoName = createVideoWriter("vistest/model", cameraProperties)
    else:
        videoWriter, videoName = createVideoWriter(f"{dateStr}/process/modelAID", cameraProperties)

    # Pyrender setup (create a Pyrender scene)
    scene = pyrender.Scene()

    # Track the node added for each mesh for removal later
    for rowIndex, data in dataCSV.iterrows():
        roll = np.radians(data['roll'])
        pitch = np.radians(data['pitch'])
        button = int(data['button'])

        T = transMatrix(roll,pitch) # Calculate transformation matrix

        # Load the appropriate mesh based on the button state
        if button == 0:  # IDLE
            mesh = trimesh.load_mesh("3dassets/pipette_up.obj")
            buttonTEXT = "Idle"
        elif button == 1:  # SUCK
            mesh = trimesh.load_mesh("3dassets/pipette_down.obj")
            buttonTEXT = "Sucking"
        elif button == 10: # RELEASE
            mesh = trimesh.load_mesh("3dassets/pipette_up.obj")
            buttonTEXT = "Releasing"
        elif button == 11: # IDLE
            mesh = trimesh.load_mesh("3dassets/pipette_down.obj")
            buttonTEXT = "Idle"

        img_bgr, mesh_node = scenePyRender(scene,mesh,T, cameraProperties) # Scene Renderer

        # Add text to the frame
        cv.putText(img_bgr, f"Pipette Controller", (0, 25), cv.FONT_HERSHEY_DUPLEX,
                   0.5, (0, 0, 0), 1, cv.LINE_AA)

        cv.putText(img_bgr, f"Roll: {round(data['roll'], 2)}, Pitch: {round(data['pitch'], 2)}, Button Pressed: {buttonTEXT}",
                   (0, 50), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        # Write the frame to the video file
        videoWriter.write(img_bgr)

        # Remove the mesh node from the scene after rendering the frame
        scene.remove_node(mesh_node)

    # Release the video writer after all frames are written
    videoWriter.release()

    return videoName

################################ GPU Video ############################################ 
def sceneo3d(vis, displayObject, T):
    """Function to render a mesh on a scene using a 4x4 matrix

    Args:
        scene (pyrender.Scene): Scene for showing object
        mesh (trimesh): Object mesh
        T : 4x4 Transformation Matrix

    Returns:
        img_bgr, mesh_node: OpenCV frame, mesh_node
    """
    # Clear geometry for next frame (since changing frames)
    vis.clear_geometries()

    displayObject.transform(T) # Apply previous transformation to the chosen mesh so that smooth animation

    # Add the object to the visualizer
    vis.add_geometry(displayObject)
    vis.update_geometry(displayObject)
    vis.poll_events()
    vis.update_renderer()

    # Capture frame and write to video
    img = vis.capture_screen_float_buffer(False)
    img_np = (np.asarray(img) * 255).astype(np.uint8)
    img_bgr = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)
    return img_bgr

def visualizer3dSCPVideoGPU(filename: str, cameraProperties: list,  dateStr: str = "", test: bool = False):
    dataCSV = pd.read_csv(filename, header = None)
    dataCSV.columns = ['roll', 'pitch', 'button', 'a', 'b', 'c']
    
    if test:
        videoWriter, videoName = createVideoWriter("vistest/model", cameraProperties)
    else:
        videoWriter, videoName = createVideoWriter(f"{dateStr}/process/model", cameraProperties)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=int(cameraProperties[0]), height=int(cameraProperties[1]))

    for _, data in dataCSV.iterrows():
        roll= np.radians(data['roll'])
        pitch = np.radians(data['pitch'])
        button = data['button']

        T = transMatrix(roll,pitch) # Calculate transformation matrix

        if button == 0: # button down
            displayObject = o3d.io.read_triangle_mesh("3dassets/pipette_up.obj")

        elif button == 1: # button down
            displayObject = o3d.io.read_triangle_mesh("3dassets/pipette_down.obj")

        img_bgr = sceneo3d(vis,displayObject,T)

        cv.putText(img_bgr, f"Single Channel Pippette",
        (0, 25), cv.FONT_HERSHEY_DUPLEX,
        FONT_SIZE/2, (0,0,0), FONT_THICKNESS, cv.LINE_AA)

        cv.putText(img_bgr, f"Roll: {round(data['roll'],2)}, Pitch: {round(data['pitch'],2)}, Button Pressed: {bool(data['button'])}",
        (0, 50), cv.FONT_HERSHEY_DUPLEX,
        FONT_SIZE/2, (0,0,0), FONT_THICKNESS, cv.LINE_AA)

        videoWriter.write(img_bgr)

    # Release the video writer and destroy the visualizer window 
    vis.destroy_window()
    videoWriter.release()

    return videoName

def visualizer3dAIDVideoGPU(filename: str, cameraProperties: list,  dateStr: str = "", test: bool = False):
    dataCSV = pd.read_csv(filename, header = None)
    dataCSV.columns = ['a', 'b', 'c','roll', 'pitch', 'button']
    
    if test:
        videoWriter, videoName = createVideoWriter("vistest/model", cameraProperties)
    else:
        videoWriter, videoName = createVideoWriter(f"{dateStr}/process/model", cameraProperties)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=int(cameraProperties[0]), height=int(cameraProperties[1]))

    for _, data in dataCSV.iterrows():
        roll = np.radians(data['roll'])
        pitch = np.radians(data['pitch'])
        button = int(data['button'])

        T = transMatrix(roll,pitch) # Calculate transformation matrix

        if button == 0: # no button
            displayObject = o3d.io.read_triangle_mesh("3dassets/pipette_up.obj")
            buttonTEXT = "Idle"

        elif button == 1: # suck button
            displayObject = o3d.io.read_triangle_mesh("3dassets/pipette_down.obj")
            buttonTEXT = "Suck"
        
        elif button == 10: # release button
            displayObject = o3d.io.read_triangle_mesh("3dassets/pipette_up.obj")
            buttonTEXT = "Release"

        elif button == 11: # borken
            displayObject = o3d.io.read_triangle_mesh("3dassets/pipette_down.obj")
            buttonTEXT = "Idle"

        img_bgr = sceneo3d(vis,displayObject,T)

        # Add text to the frame
        cv.putText(img_bgr, f"Pipette Controller", (0, 25), cv.FONT_HERSHEY_DUPLEX,
                   0.5, (0, 0, 0), 1, cv.LINE_AA)

        cv.putText(img_bgr, f"Roll: {round(data['roll'], 2)}, Pitch: {round(data['pitch'], 2)}, Button Pressed: {buttonTEXT}",
                   (0, 50), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        videoWriter.write(img_bgr)

    # Release the video writer and destroy the visualizer window 
    vis.destroy_window()
    videoWriter.release()

    return videoName
################################ CSV Test File Generators ############################################ 
def generate_random_transform_csv(num_samples):
    """Used to generate random csv to test above code.

    Args:
        num_samples (int): How many samples of data do you want to take

    Returns:
        str: name of the csv made
    """
    date = datetime.now()
    dateStr = date.strftime("%d%m%y %H%M%S")
    filename = f"testCSV/{dateStr}.csv"
    
    # Initialize lists to store the values
    roll_values = []
    pitch_values = []
    button1 = np.random.choice([0,1]) 
    button2 = random.choice(['00', '01', '10']) 
    
    # Initialize random starting roll and pitch
    roll = np.random.uniform(0, 360)
    pitch = np.random.uniform(0, 360)
    
    # Append the initial roll and pitch values
    roll_values.append(roll)
    pitch_values.append(pitch)
    
    for _ in range(1, num_samples):
        # Randomly change the roll and pitch by a value between 0 and 2 degrees
        roll_change = np.random.uniform(0, 1)
        pitch_change = np.random.uniform(0, 1)
        
        # Apply the change to the previous values (can increase or decrease)
        roll += np.random.choice([-1, 1]) * roll_change
        pitch += np.random.choice([-1, 1]) * pitch_change
        
        # Ensure roll and pitch stay within bounds (0-360 degrees)
        roll = np.clip(roll, 0, 360)
        pitch = np.clip(pitch, 0, 360)
        
        # Append the new values to the lists
        roll_values.append(roll)
        pitch_values.append(pitch)

    # Create the DataFrame
    data = {
        'rollSCP': roll_values,
        'pitchSCP': pitch_values,
        'buttonSCP': button1,
        'rollAID': roll_values,
        'pitchAID': pitch_values,
        'buttonAID': button2
    }
    df = pd.DataFrame(data)
    
    # Write to CSV without headers
    df.to_csv(filename, index=False, header=False)
    print(f"{num_samples} samples written to {filename}")
    return filename

def generate_sweep_csv(num_per_sweep: int = 360, toggle_interval: int = 90):
    date = datetime.now().strftime("%d%m%y_%H%M%S")
    filename = f"testCSV/{date}.csv"

    # Sweep roll
    roll_1 = np.linspace(0, 360, num=num_per_sweep)
    pitch_1 = np.zeros(num_per_sweep)

    # Sweep pitch
    roll_2 = np.zeros(num_per_sweep)
    pitch_2 = np.linspace(0, 360, num=num_per_sweep)

    # Concatenate
    roll = np.concatenate((roll_1, roll_2))
    pitch = np.concatenate((pitch_1, pitch_2))

    # Generate flag that toggles every 90 samples
    scpButtonList = []
    scpButton = 0
    for i in range(0, len(roll)):
        if i % toggle_interval == 0:
            scpButton = 1 - scpButton
        scpButtonList.append(scpButton)

    idle = np.full(int(num_per_sweep/2),'00') # 00
    suck = np.full(int(num_per_sweep/2),'01') # 01
    release = np.full(int(num_per_sweep/2),'10') # 10
    broke = np.full(int(num_per_sweep/2),'11') # 11
    aidButtonList=list(np.concatenate((idle,suck,release,broke)))

    # Create DataFrame
    data = {
        'rollSCP': roll,
        'pitchSCP': pitch,
        'buttonSCP': scpButtonList,
        'rollAID': roll,
        'pitchAID': pitch,
        'buttonAID': aidButtonList
    }
    df = pd.DataFrame(data)

    # Save to CSV without headers or index
    df.to_csv(filename, index=False, header=False)
    print(f"CSV written to: {filename}")

    return filename

if torch.cuda.is_available():
    visualizer3dSCPVideo = visualizer3dSCPVideoGPU
    visualizer3dAIDVideo = visualizer3dAIDVideoGPU
    print("================")
    print("Visualise on GPU")
else:
    visualizer3dSCPVideo = visualizer3dSCPVideoCPU
    visualizer3dAIDVideo = visualizer3dAIDVideoCPU
    print("================")
    print("Visualise on CPU")

if __name__=="__main__":
    # print("Using Mesa OpenGL:", os.path.exists("opengl32.dll"))

    cap = cv.VideoCapture(0)
    
    if not cap.isOpened:
        print("Camera brokey")
        exit()

    cameraProperties = getCameraProperties(cap)

    cap.release()

    # Test visualisation
    # randomTestFile = generate_random_transform_csv(150)
    randomTestFile = generate_sweep_csv(40,20)
    
    pyrendertime = time.time()
    animatedVideo = visualizer3dSCPVideoCPU(randomTestFile, cameraProperties, test = True)
    pyrendertime = time.time() - pyrendertime

    o3drendertime = time.time()
    animatedVideo = visualizer3dSCPVideoGPU(randomTestFile, cameraProperties, test = True)
    o3drendertime = time.time() - o3drendertime

    print(pyrendertime,o3drendertime)




