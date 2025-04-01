#include <WiFi.h>
#include <HTTPClient.h>

const char *ssid = "ESP32_AP";  // SSID of the Access Point
const char *password = "123456789";  // Password for the Access Point
const char *serverName = "http://192.168.4.1/send";  // Server IP and POST endpoint

int i = 0;

void setup() {
  Serial.begin(115200);

  // Connect to the Access Point
  WiFi.begin(ssid, password);

  // Wait for the connection to establish
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }

  Serial.println("Connected to WiFi");
}

void loop() {
  // Nothing to do in the loop
  // Create the HTTPClient instance
  HTTPClient http;

  // Start the HTTP request
  http.begin(serverName);  // Specify the server URL

  // Set the content type for the POST request
  http.addHeader("Content-Type", "text/plain");

  // Send the POST request with the "Hello World" message in the body
  String payload = "Slave 2:" + String(i);
  int httpCode = http.POST(payload);  // Send the POST request

  if (httpCode > 0) {
    // If the request was successful, print the response
    Serial.printf("HTTP Response code: %d\n", httpCode);
    String response = http.getString();
    Serial.println("Server response: " + response);
  } else {
    // If there was an error, print the error
    Serial.printf("Error on HTTP request: %s\n", http.errorToString(httpCode).c_str());
  }

  // Close the HTTP connection
  http.end();
  i+=1;
  delay(100);
}
