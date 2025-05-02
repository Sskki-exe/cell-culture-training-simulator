#include <WiFi.h>
#include <WebServer.h>

const char *ssid = "ESP32_AP";  // Access Point SSID
const char *password = "123456789";  // Access Point Password

WebServer server(80);  // Web server running on port 80

// Handle POST request to "/send"
void handlePost() {
  if (server.hasArg("plain")) {  // Check if the body has content
    String message = server.arg("plain");  // Get the message from the body
    Serial.println("Received message: " + message);
    server.send(200, "text/plain", "Message received: " + message);  // Respond back to the client
  } else {
    server.send(400, "text/plain", "No message received");  // Error if no message
  }
}

void setup() {
  Serial.begin(115200);

  // Set up the ESP32 as an Access Point
  WiFi.softAP(ssid, password);
  Serial.println("Access Point Started");

  // Print the IP address of the Access Point
  Serial.println("IP Address: ");
  Serial.println(WiFi.softAPIP());

  // Define a route for the HTTP POST request
  server.on("/send", HTTP_POST, handlePost);

  // Start the HTTP server
  server.begin();
}

void loop() {
  server.handleClient();  // Handle incoming HTTP requests
}
