import pandas as pd
import cv2 as cv
import numpy as np
import open3d as o3d
from camera import createVideoWriter, getCameraProperties
from datetime import datetime

def visualizerVideo(filename: str, cameraProperties: list):
    dataCSV = pd.read_csv(filename, header = None)
    dataCSV.columns = ['roll','pitch','button']
    videoWriter, videoName = createVideoWriter("vistest/model", cameraProperties)

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

        # rollDiff = roll-rollOld
        # pitchDiff = pitch-pitchOld

        # print(rollDiff)
        # print(pitchDiff)

        # Rx = np.array([
        #     [1,0,0],
        #     [0, np.cos(rollDiff), -np.sin(rollDiff)],
        #     [0, np.sin(rollDiff), np.cos(rollDiff)]
        # ]) # Rotation using roll

        # Ry = np.array([
        #     [np.cos(pitchDiff), 0, np.sin(pitchDiff)],
        #     [0, 1, 0],
        #     [-np.sin(pitchDiff), 0, np.cos(pitchDiff)]
        # ]) # Rotation using pitch

        R = Rx @ Ry # Rotation combination
        T = np.eye(4) # Transformation matrix
        T[:3,:3] = R
        print(T)

        if button == 0: # button down
            displayObject = o3d.io.read_triangle_mesh("visualiser/pipette_up.obj")

        elif button == 1: # button down
            displayObject = o3d.io.read_triangle_mesh("visualiser/pipette_down.obj")

        # Clear geometry for next frame (since changing frames)
        vis.clear_geometries()

        # displayObject.transform(np.eye(4)) # Apply previous transformation to the chosen mesh so that smooth animation
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
        videoWriter.write(img_bgr)

        rollOld = roll
        pitchOld = pitch

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
    # date = datetime.now()
    # dateStr = date.strftime("%d%m%y %H%M%S")
    # filename = f"testCSV/{dateStr}"
    # data = {
    #     'roll': np.random.uniform(0, 360, size=num_samples),
    #     'pitch': np.random.uniform(0, 360, size=num_samples),
    #     'object_flag': np.random.randint(0, 2, size=num_samples)  # 0 or 1
    # }
    # df = pd.DataFrame(data)
    # df.to_csv(filename, index=False, header=False)  # No headers
    # print(f"{num_samples} samples written to {filename}")
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

if __name__=="__main__":
    cap = cv.VideoCapture(0)
    
    if not cap.isOpened:
        print("Camera brokey")
        exit()

    cameraProperties = getCameraProperties(cap)

    cap.release()

    randomTestFile = generate_random_transform_csv(500)

    animatedVideo = visualizerVideo(randomTestFile, cameraProperties)

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