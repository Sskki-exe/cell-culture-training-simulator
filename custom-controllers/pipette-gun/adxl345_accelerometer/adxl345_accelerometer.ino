#include <Wire.h>  // Wire library - used for I2C communication

int MPU6050 = 0x68; // The MPU-6050 sensor I2C address

float X_out, Y_out, Z_out;  // Accelerometer outputs
float roll, pitch, rollF, pitchF = 0;

const int button_pin = 2;

void setup() {
  Serial.begin(9600); // Initiate serial communication for printing the results on the Serial monitor
  
  pinMode(button_pin, INPUT);
  Wire.begin(1, 0); // SDA SCL Initiate the Wire library

  // Initialize MPU-6050
  Wire.beginTransmission(MPU6050); // Start communicating with the device
  Wire.write(0x6B); // Access the power management register (0x6B)
  Wire.write(0);    // Wake up the MPU-6050 by writing 0 to the power management register
  Wire.endTransmission();
  delay(10);
}

void loop() {
  // === Read accelerometer data === //
  Wire.beginTransmission(MPU6050);
  Wire.write(0x3B); // Start reading from register 0x3B (ACCEL_XOUT_H)
  Wire.endTransmission(false);
  Wire.requestFrom(MPU6050, 6, true); // Read 6 bytes of data

  // Read accelerometer values (16-bit, 2 bytes per axis)
  X_out = (Wire.read() << 8 | Wire.read()); // X-axis value
  Y_out = (Wire.read() << 8 | Wire.read()); // Y-axis value
  Z_out = (Wire.read() << 8 | Wire.read()); // Z-axis value

  // Convert to 'g' by dividing by 16384 (MPU-6050 default sensitivity)
  X_out = X_out / 16384.0;
  Y_out = Y_out / 16384.0;
  Z_out = Z_out / 16384.0;

  // Calculate Roll and Pitch (rotation around X-axis, rotation around Y-axis)
  roll = atan(Y_out / sqrt(pow(X_out, 2) + pow(Z_out, 2))) * 180 / PI;
  pitch = atan(-1 * X_out / sqrt(pow(Y_out, 2) + pow(Z_out, 2))) * 180 / PI;

  // Low-pass filter
  rollF = 0.94 * rollF + 0.06 * roll;
  pitchF = 0.94 * pitchF + 0.06 * pitch;

  // Output roll, pitch, and button status to Serial Monitor
  Serial.print(rollF);
  Serial.print("/");
  Serial.print(pitchF);
  Serial.print("/");
  Serial.println(!digitalRead(button_pin));
}
