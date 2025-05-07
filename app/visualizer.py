import pandas as pd
import cv2 as cv
import numpy as np
import open3d as o3d
from camera import createVideoWriter, getCameraProperties
from datetime import datetime

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def visualizerVideo(filename: str, cameraProperties: list,  dateStr: str = "", test: bool = False):
    dataCSV = pd.read_csv(filename, header = None)
    dataCSV.columns = ['roll','pitch','button']
    if test:
        videoWriter, videoName = createVideoWriter("vistest/model", cameraProperties)
    else:
        videoWriter, videoName = createVideoWriter(f"{dateStr}/process/model", cameraProperties)

    # buttonUp = o3d.io.read_triangle_mesh("visualiser/pipette_up.obj")
    # buttonDown = o3d.io.read_triangle_mesh("visualiser/pipette_down.obj")

    vis = o3d.visualization.Visualizer()
    print(cameraProperties)
    vis.create_window(visible=False, width=int(cameraProperties[0]), height=int(cameraProperties[1]))

    T = np.eye(4)
    # rollOld = 0
    # pitchOld = 0

    for rowIndex, data in dataCSV.iterrows():
        roll= np.radians(data['roll'])
        pitch = np.radians(data['pitch'])
        button = data['button']

        Rx = np.array([
            [1,0,0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ]) # Rotation using roll

        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ]) # Rotation using pitch

        R = Rx @ Ry # Rotation combination
        T = np.eye(4) # Transformation matrix
        T[:3,:3] = R
        # print(T)

        if button == 0: # button down
            displayObject = o3d.io.read_triangle_mesh("visualiser/pipette_up.obj")

        elif button == 1: # button down
            displayObject = o3d.io.read_triangle_mesh("visualiser/pipette_down.obj")

        # print("Textures loaded:", len(displayObject.textures) > 0)

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

        cv.putText(img_bgr, f"Roll: {round(data['roll'],2)}, Pitch: {round(data['pitch'],2)}, Button Pressed: {bool(data['button'])}",
        (0, 50), cv.FONT_HERSHEY_DUPLEX,
        FONT_SIZE/2, (0,0,0), FONT_THICKNESS, cv.LINE_AA)

        videoWriter.write(img_bgr)

    # Release the video writer and destroy the visualizer window 
    vis.destroy_window()
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
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened:
        print("Camera brokey")
        exit()

    cameraProperties = getCameraProperties(cap)

    cap.release()

    # randomTestFile = generate_random_transform_csv(500)
    randomTestFile = generate_sweep_csv()

    animatedVideo = visualizerVideo(randomTestFile, cameraProperties, test = True)

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


