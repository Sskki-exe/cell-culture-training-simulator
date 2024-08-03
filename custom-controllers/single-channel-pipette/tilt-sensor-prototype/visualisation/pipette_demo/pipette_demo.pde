import processing.serial.*;
import java.awt.event.KeyEvent;
import java.io.IOException;

PShape model_pipette_up;
PShape model_pipette_down;
Serial myPort;

String data="";
float roll, pitch;
boolean button_state;

void setup() {
  size (960, 640, P3D);
  myPort = new Serial(this, "/dev/ttyUSB0", 9600); // starts the serial communication
  myPort.bufferUntil('\n');
  
  model_pipette_up = loadShape("render_models/pipette_up.obj");
  model_pipette_down = loadShape("render_models/pipette_down.obj");
  shapeMode(CENTER);
}

void draw() {
  translate(width/2, height/2, 0);
  background(33);
  lights();
  textSize(22);
  text("Roll: " + int(roll) + "     Pitch: " + int(pitch), -100, 265);

  // Rotate the object
  rotateX(radians(90+roll));
  rotateY(radians(pitch));
  rotateZ(radians(-90));
  
  scale(15);
  
  if (button_state) {
    shape(model_pipette_up);
  } else {
    shape(model_pipette_down);
  }
}

// Read data from the Serial Port
void serialEvent (Serial myPort) { 
  // reads the data from the Serial Port up to the character '.' and puts it into the String variable "data".
  data = myPort.readStringUntil('\n');

  // if you got any bytes other than the linefeed:
  if (data != null) {
    data = trim(data);
    // split the string at "/"
    String items[] = split(data, '/');
    if (items.length > 1) {

      //--- Roll,Pitch in degrees
      roll = float(items[0]);
      pitch = float(items[1]);
      button_state = int(items[2])==0;
    }
  }
}
