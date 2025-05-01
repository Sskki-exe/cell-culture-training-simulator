#include <WiFi.h>       // Wifi library
#include <HTTPClient.h> // HTTP communication library

/* Wireless Config */
const char *ssid = "ESP32_AP";       // Hub SSID
const char *password = "123456789";  // Hub Password
WebServer server(80);                // Use port 80 

/* Handle Sensor Messages */
void handlePost() {
  // Check if message empty
  if (!server.hasArg("plain")) {
    server.send(400, "text/plain", "No message received");
    return
  }

  String message = server.arg("plain");
  Serial.println(message);

  // Send acknowledgement to sensor
  server.send(200, "text/plain", "Message received: " + message);
}

void setup() {
  /* Start Serial communication for debugging purposes*/
  Serial.begin(115200);
  Serial.setTimeout(1); 
  
  /* Start HTTP Server (for sensors to connect to) */
  WiFi.softAP(ssid, password);
  Serial.println("Access Point Started");
  Serial.println("IP Address: ");
  Serial.println(WiFi.softAPIP());
  server.on("/send", HTTP_POST, handlePost);  // Set request route
  server.begin();                             // Start HTTP server
}

void loop() {
  server.handleClient();  // Handle incoming HTTP requests
}
