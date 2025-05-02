void setup() {
  /* Start Serial communication for debugging purposes*/
  Serial.begin(115200);
  Serial.setTimeout(1); 
}

void loop() {
  Serial.println("Hello World!");
}
