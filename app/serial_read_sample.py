import serial   # Read Serial
import random

# ========== USB Config ========== #
COMPORT = 'COM3'    # Specify ESP32 COM port
# ================================ #

# arduino = serial.Serial(port=COMPORT, baudrate=115200, timeout=.1)  # Define Serial Settings

# Read Serial function
def read_hub_serial():
    # data = arduino.readline().decode('utf-8').strip()  # Decode bytes to str and strip newline
    
    # For simulation
    rollF = round(random.uniform(-90, 90), 2)     # Simulated roll in degrees
    pitchF = round(random.uniform(-90, 90), 2)    # Simulated pitch in degrees
    button_state = random.choice([0, 1])          # 1 = pressed, 0 = not pressed
    data = f"Pipette_1:{rollF}/{pitchF}/{button_state}"
    return data # pipette1:

# Sample main loop
while True: 
    num = input("Press [Enter] to take a reading")  # Janky keyboard input solution
    print(read_hub_serial())
