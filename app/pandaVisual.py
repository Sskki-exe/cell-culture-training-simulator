import pandas as pd
import cv2 as cv
import numpy as np
import random
import trimesh
from camera import createVideoWriter, getCameraProperties
from datetime import datetime
import time
from panda3d.core import *
loadPrcFileData("", """window-type offscreen\n""")
loadPrcFileData('', 'load-display p3tinydisplay')
from direct.showbase.ShowBase import ShowBase

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

    # flip1 =  np.array([
    #     [1, 0, 0],
    #     [0, 0, 1],
    #     [0, -1, 0]
    # ]) 

    flip1 =  1.8 * np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ]) 

    # Combine roll and pitch rotations
    R = flip1 @ Ry @ Rx 
    T = np.eye(4)
    T[:3, :3] = R

    return T

def np2panda(T):
    """Convert from numpy T matrix to panda T matrix

    Args:
        T (4x4 np.array): 4x4 Homogenous Tranformation matrix

    Returns:
        Mat4 : Panda format for matrix
    """
    T = np.array(T, dtype=np.float32).reshape(4, 4)

    # Convert to row-major (Panda expects row-major)
    pandaT = LMatrix4f()
    for i in range(4):
        for j in range(4):
            pandaT.set_cell(i, j, T[i, j])

    return Mat4(pandaT)

################################ CPU Video ############################################ 
class PandaRenderer(ShowBase):
    def __init__(self, cameraProperties):
        # Initialize ShowBase without opening a visible window
        ShowBase.__init__(self, windowType='offscreen')
        
        self.width, self.height = int(cameraProperties[0]), int(cameraProperties[1])
        
        fb_props = FrameBufferProperties()
        fb_props.set_rgb_color(True)
        fb_props.set_alpha_bits(8)
        fb_props.set_depth_bits(24)
        
        win_props = WindowProperties.size(self.width, self.height)


        # Create offscreen buffer directly, no temporary window needed
        self.buffer = self.graphicsEngine.make_output(
            self.pipe, "offscreen buffer", -2,
            fb_props, win_props,
            GraphicsPipe.BF_refuse_window,
            None,  # Use default GSG
            None
        )
        
        if self.buffer is None:
            raise RuntimeError("Failed to create offscreen buffer")
        
        # Create texture and attach to buffer
        self.tex = Texture()
        self.buffer.add_render_texture(self.tex, GraphicsOutput.RTMCopyRam)
        
        # Create camera
        self.cam_node = self.makeCamera(self.buffer)
        lens = PerspectiveLens()
        lens.set_fov(90)
        lens.set_aspect_ratio(self.width / self.height)
        self.cam_node.node().set_lens(lens)
        
        # Position camera so it can see your model
        self.cam_node.set_pos(0, -30, 0)
        self.cam_node.look_at(0, 0, 20)
        
        # Set up light
        dlight = DirectionalLight("dlight")
        dlight.set_color((1, 1, 1, 1))
        dlight.set_direction((0, 0, -1))
        dlnp = self.render.attach_new_node(dlight)
        self.render.set_light(dlnp)

        # Ambient light
        alight = AmbientLight("alight")
        alight.set_color((0.3, 0.3, 0.3, 1))
        alnp = self.render.attach_new_node(alight)
        self.render.set_light(alnp)
                
        self.mesh_node = None
        self.rotationFix = Mat4.rotate_mat(180, LVector3(0, 1, 0))
    
    def render_mesh(self, mesh_path, T):
        # Load mesh, apply transform, render, and return numpy array
        if self.mesh_node:
            self.mesh_node.remove_node()
        
        model = self.loader.load_model(mesh_path)
        model.reparent_to(self.render)

        # axis = self.loader.load_model("models/zup-axis")  # built-in axis model
        # axis.set_scale(5)  # scale as needed
        # axis.reparent_to(self.render)
        
        TPanda = np2panda(T)  # Convert numpy matrix to Panda Mat4
        model.set_mat(self.rotationFix*TPanda)
        
        self.mesh_node = model
        
        self.graphicsEngine.render_frame()
        
        # Get image data from texture
        img = self.tex.get_ram_image_as("RGB")
        # Convert to numpy array (bytes -> numpy)
        img_np = np.frombuffer(img.get_data(), dtype=np.uint8)
        img_np = img_np.reshape((self.height, self.width, 3))
        img_np = np.flipud(img_np)  # Flip vertical to match usual image coords
        img_bgr = cv.cvtColor(img_np, cv.COLOR_RGB2BGR)

        return img_bgr

def visualizer3dSCPVideoPanda(filename, cameraProperties, renderer, dateStr="", test=False):
    dataCSV = pd.read_csv(filename, header = None)
    dataCSV.columns = ['roll', 'pitch', 'button', 'a', 'b', 'c']
    
    if test:
        videoWriter, videoName = createVideoWriter("vistest/model", cameraProperties)
    else:
        videoWriter, videoName = createVideoWriter(f"{dateStr}/process/model", cameraProperties)

    for _, row in dataCSV.iterrows():
        roll = row['roll']
        pitch = row['pitch']
        button = row['button']

        T = transMatrix(np.deg2rad(roll),np.deg2rad(pitch))
        
        # Load the mesh based on button state
        if button == 0:
            mesh = "3dassets/pipette_up.obj"
        elif button == 1:
            mesh = "3dassets/pipette_down.obj"
        
        # if button == 0:
        #     mesh = trimesh.load_mesh("3dassets/pipette_up.obj")
        # elif button == 1:
        #     mesh = trimesh.load_mesh("3dassets/pipette_down.obj")

        img_bgr = renderer.render_mesh(mesh, T)

        cv.putText(img_bgr, f"Single Channel Pipette", (0, 25), cv.FONT_HERSHEY_DUPLEX,
                   0.5, (0, 0, 0), 1, cv.LINE_AA)

        cv.putText(img_bgr, f"Roll: {round(row['roll'], 2)}, Pitch: {round(row['pitch'], 2)}, Button Pressed: {bool(row['button'])}",
                   (0, 50), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        videoWriter.write(img_bgr)

    videoWriter.release()
    return videoName

def visualizer3dAIDVideoPanda(filename, cameraProperties, renderer, dateStr="", test=False):
    dataCSV = pd.read_csv(filename, header = None)
    dataCSV.columns = ['roll', 'pitch', 'button', 'a', 'b', 'c']
    # renderer = PandaRenderer(cameraProperties)
    
    if test:
        videoWriter, videoName = createVideoWriter("vistest/model", cameraProperties)
    else:
        videoWriter, videoName = createVideoWriter(f"{dateStr}/process/model", cameraProperties)

    for _, row in dataCSV.iterrows():
        roll = row['roll']
        pitch = row['pitch']
        button = row['button']

        T = transMatrix(np.deg2rad(roll),np.deg2rad(pitch))
        # Load the mesh based on button state        
        if button == 0: # no button
            displayObject = "3dassets/pipette_default.obj"
            buttonTEXT = "Idle"

        elif button == 1: # suck button
            displayObject = "3dassets/pipette_aspirate.obj"
            buttonTEXT = "Aspirate"
        
        elif button == 10: # release button
            displayObject = "3dassets/pipette_dispense.obj"
            buttonTEXT = "Dispense"

        elif button == 11: # borken
            displayObject = "3dassets/pipette_abuse.obj"
            buttonTEXT = "Idle"

        img_bgr = renderer.render_mesh(displayObject, T)

        cv.putText(img_bgr, f"Pipette Aid", (0, 25), cv.FONT_HERSHEY_DUPLEX,
                   0.5, (0, 0, 0), 1, cv.LINE_AA)

        cv.putText(img_bgr, f"Roll: {round(row['roll'], 2)}, Pitch: {round(row['pitch'], 2)}, Button Pressed: {buttonTEXT}",
                   (0, 50), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

        videoWriter.write(img_bgr)

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

if __name__=="__main__":
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened:
        print("Camera brokey")
        exit()

    cameraProperties = getCameraProperties(cap)

    cap.release()

    # Test visualisation
    # randomTestFile = generate_random_transform_csv(150)
    randomTestFile = generate_sweep_csv()
    renderer = PandaRenderer(ca)
    pyrendertime = time.time()
    animatedVideo = visualizer3dSCPVideoPanda(randomTestFile, cameraProperties, renderer, test = True)
    pyrendertime = time.time() - pyrendertime
    print(pyrendertime)

    # o3drendertime = time.time()
    # animatedVideo = visualizer3dSCPVideoGPU(randomTestFile, cameraProperties, test = True)
    # o3drendertime = time.time() - o3drendertime

    # print(pyrendertime,o3drendertime)

    # pipes = GraphicsPipeSelection.get_global_ptr()
    # for i in range(pipes.get_num_pipe_types()):
    #     print(pipes.get_pipe_type(i).get_name())


