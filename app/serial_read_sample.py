import serial   # Read Serial
import random
import time

# ========== USB Config ========== #
COMPORT = 'COM3'    # Specify ESP32 COM port
# ================================ #

# arduino = serial.Serial(port=COMPORT, baudrate=115200, timeout=.1)  # Define Serial Settings

# Read Serial function
def read_hub_serial():
    # arduino.write(b'this can be any text with a newline\n')
    # data1 = arduino.readline().decode('utf-8').strip()  # Decode bytes to str and strip newline
    # data = arduino.readline()

    # # For simulation
    rollF = round(random.uniform(-90, 90), 2)     # Simulated roll in degrees
    pitchF = round(random.uniform(-90, 90), 2)    # Simulated pitch in degrees
    button_state1 = random.choice([0,1]) 
    button_state2 = random.choice(['00', '01', '10']) 
    data1 = f"Pipette_1:{rollF}/{pitchF}/{button_state1}"
    data2 = f"+Aid_1:{rollF}/{pitchF}/{button_state2}"
    data = data1 + data2
    return data

# Sample main loop
if __name__ == "__main__":
    while True: 
        print(read_hub_serial())
        time.sleep(1/30)
