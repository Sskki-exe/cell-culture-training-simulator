import pandas as pd
import cv2 as cv
import numpy as np
import os
import pyrender
import trimesh
# os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
# os.environ["OPEN3D_CPU_RENDERING"] = "true"
import open3d as o3d
from camera import createVideoWriter, getCameraProperties
from datetime import datetime




MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def visualizer3dVideo(filename: str, cameraProperties: list, dateStr: str = "", test: bool = False):
    # Read the CSV file with transformation data
    dataCSV = pd.read_csv(filename, header=None)
    dataCSV.columns = ['roll', 'pitch', 'button']
    
    # Set up video writer based on test flag
    if test:
        videoWriter, videoName = createVideoWriter("vistest/model", cameraProperties)
    else:
        videoWriter, videoName = createVideoWriter(f"{dateStr}/process/model", cameraProperties)

    # Pyrender setup (create a Pyrender scene)
    scene = pyrender.Scene()

    # Track the node added for each mesh for removal later
    mesh_nodes = []

    for rowIndex, data in dataCSV.iterrows():
        roll = np.radians(data['roll'])
        pitch = np.radians(data['pitch'])
        button = data['button']

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

        # Load the appropriate mesh based on the button state
        if button == 0:  # button down
            mesh = trimesh.load_mesh("3dassets/pipette_up.obj")
        elif button == 1:  # button down
            mesh = trimesh.load_mesh("3dassets/pipette_down.obj")

        # Apply the transformation to the mesh
        mesh.apply_transform(T)

        # Create a Pyrender mesh from the Trimesh object
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)

        # Create a Pyrender node for the mesh and add it to the scene
        mesh_node = scene.add(pyrender_mesh)
        mesh_nodes.append(mesh_node)

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

        # Add text to the frame
        cv.putText(img_bgr, f"Single Channel Pipette", (0, 25), cv.FONT_HERSHEY_DUPLEX,
                   0.5, (0, 0, 0), 2, cv.LINE_AA)

        cv.putText(img_bgr, f"Roll: {round(data['roll'], 2)}, Pitch: {round(data['pitch'], 2)}, Button Pressed: {bool(data['button'])}",
                   (0, 50), cv.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 2, cv.LINE_AA)

        # Write the frame to the video file
        videoWriter.write(img_bgr)

        # Remove the mesh node from the scene after rendering the frame
        scene.remove_node(mesh_node)

    # Release the video writer after all frames are written
    videoWriter.release()

    return videoName




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
    button = np.random.randint(0, 2, size=num_samples)  # 0 or 1
    
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
        'roll': roll_values,
        'pitch': pitch_values,
        'button': button
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
    object_flag = []
    current_flag = 0
    for i in range(0, len(roll)):
        if i % toggle_interval == 0:
            current_flag = 1 - current_flag
        object_flag.append(current_flag)

    # Create DataFrame
    data = {
        'roll': roll,
        'pitch': pitch,
        'object_flag': object_flag
    }
    df = pd.DataFrame(data)

    # Save to CSV without headers or index
    df.to_csv(filename, index=False, header=False)
    print(f"CSV written to: {filename}")

    return filename

if __name__=="__main__":
    # print("Using Mesa OpenGL:", os.path.exists("opengl32.dll"))

    cap = cv.VideoCapture(0)
    
    if not cap.isOpened:
        print("Camera brokey")
        exit()

    cameraProperties = getCameraProperties(cap)

    cap.release()

    # randomTestFile = generate_random_transform_csv(1000)
    randomTestFile = generate_sweep_csv()

    animatedVideo = visualizer3dVideo(randomTestFile, cameraProperties, test = True)

    footage = cv.VideoCapture(animatedVideo)

    while True:
        ret, frame = footage.read()
        if ret:
            cv.imshow("frame", frame)

        if cv.waitKey(1) == ord('q'):
            break

        if not ret:
            break
    
    footage.release()

    
    # buttonUp = o3d.io.read_triangle_mesh("visualiser/pipette_up.obj")
    # buttonDown = o3d.io.read_triangle_mesh("visualiser/pipette_down.obj")

    # print("Has vertex colors:", buttonUp.has_vertex_colors())
    # print("Has triangle uvs:", buttonUp.has_triangle_uvs())
    # print("Textures:", len(buttonUp.textures)>0)

    # print("Has vertex colors:", buttonDown.has_vertex_colors())
    # print("Has triangle uvs:", buttonDown.has_triangle_uvs())
    # print("Textures:", len(buttonDown.textures)>0)


