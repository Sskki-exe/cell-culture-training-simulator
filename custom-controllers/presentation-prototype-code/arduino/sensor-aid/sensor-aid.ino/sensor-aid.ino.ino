#include <WiFi.h>       // Wifi library
#include <HTTPClient.h> // HTTP communication library
#include <Wire.h>       // I2C communication library

/* Wireless Config */
const char *ssid = "ESP32_AP";       // Hub SSID
const char *password = "123456789";  // Hub Password
const char *serverName = "http://192.168.4.1/send";  // Hub URL

/* Wiring Config */
int ADXL345 = 0x53; // Accelerometer I2C address
const int button_pinSuck = 2;
const int button_pinRelease = 3;

/* Global Variables */
float X_out, Y_out, Z_out;  // Raw XYZ data
float roll, pitch = 0;      // Noisy angles
float rollF, pitchF=0;      // Filtered angles

void setup() {
  /* Start Serial communication for debugging purposes*/
  Serial.begin(115200);

  /* Set up button */
  pinMode(button_pinSuck, INPUT_PULLUP);  // Specify button pin
  pinMode(button_pinRelease, INPUT_PULLUP);  // Specify button pin


  /* Set up accelerometer */
  Wire.begin(); // Start I2C
  Wire.beginTransmission(ADXL345);  // Connect I2C
  Wire.write(0x2D); Wire.write(8);  // Set accelerometer measurement mode
  Wire.endTransmission();
  delay(10);
  //Off-set Calibration
  //X-axis
  Wire.beginTransmission(ADXL345);
  Wire.write(0x1E); Wire.write(1);
  Wire.endTransmission();
  delay(10);
  //Y-axis
  Wire.beginTransmission(ADXL345);
  Wire.write(0x1F); Wire.write(-2);
  Wire.endTransmission();
  delay(10);
  //Z-axis
  Wire.beginTransmission(ADXL345);
  Wire.write(0x20); Wire.write(-9);
  Wire.endTransmission();
  delay(10);

  /* Connect to HTTP server (hub) */
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { // Connection attempt loop
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
}

void loop() {
  /* Read accelerometer data */
  Wire.beginTransmission(ADXL345);
  Wire.write(0x32);                                 // Request accelerometer register
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345, 6, true);               // Read accelerometer register
  X_out = ( Wire.read() | Wire.read() << 8) / 256;  // X-axis value
  Y_out = ( Wire.read() | Wire.read() << 8) / 256;  // Y-axis value
  Z_out = ( Wire.read() | Wire.read() << 8) / 256;  // Z-axis value
  // Convert XYZ to angles
  roll = atan(Y_out / sqrt(pow(X_out, 2) + pow(Z_out, 2))) * 180 / PI;
  pitch = atan(-1 * X_out / sqrt(pow(Y_out, 2) + pow(Z_out, 2))) * 180 / PI;
  // Low-pass filter
  rollF = 0.94 * rollF + 0.06 * roll;
  pitchF = 0.94 * pitchF + 0.06 * pitch;


  /* Send HTTP request (message to hub) */
  HTTPClient http;
  http.begin(serverName);                       // Set Hub URL
  http.addHeader("Content-Type", "text/plain"); // Set request type
  /* ========== Output Message ========== */
  String payload = "Aid_1:";
  payload += String(rollF) + "/";
  payload += String(pitchF) + "/";
  payload += String(!digitalRead(button_pin));
  /* ==================================== */
  int httpCode = http.POST(payload);            // Send request
  if (httpCode > 0) {                           // Confirm response from hub
    Serial.printf("HTTP Response code: %d\n", httpCode);
    String response = http.getString();
    Serial.println("Server response: " + response);
  } else {                                      // Throw error if no response
    Serial.printf("Error on HTTP request: %s\n", http.errorToString(httpCode).c_str());
  }
  http.end();                                   // End transmission
}
