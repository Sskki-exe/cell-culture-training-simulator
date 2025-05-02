import serial   # Read Serial

# ========== USB Config ========== #
COMPORT = '/dev/ttyUSB0'    # Specify ESP32 COM port
# ================================ #

arduino = serial.Serial(port=COMPORT, baudrate=115200, timeout=.1)  # Define Serial Settings

# Read Serial function
def read_hub_serial():
    data = arduino.readline()[:-3] # Remove the newline characters
    return data

# Sample main loop
while True: 
    num = input("Press [Enter] to take a reading")  # Janky keyboard input solution
    print(read_hub_serial())
