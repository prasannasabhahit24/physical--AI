/*
  Arm controller firmware (Arduino Uno / Nano / Mega) for a 4-servo arm + gripper.

  Serial protocol, 115200 baud, one ASCII line per command (see robot_arm.py):
    P                      -> PONG
    J b s e w dps          -> OK      move the 4 joint servos (degrees) at dps deg/s; replies when finished
    G deg                  -> OK      gripper servo angle
    Q                      -> S <gripper_current_adc> <gripper_deg>
    E                      -> OK      emergency stop: servos detached, power relay off
    R                      -> OK      re-enable (only if the hardware E-stop button is released)
  While stopped, J and G reply "ESTOP".

  Wiring
    D3  base servo      D5  shoulder servo     D6  elbow servo
    D9  wrist servo     D10 gripper servo
    A0  gripper current sense (shunt / ACS712 on the gripper servo supply)
    D2  hardware E-stop button to GND (normally closed button recommended: wire so that
        a broken wire also reads as "pressed" - here the pin is INPUT_PULLUP, pressed = LOW)
    D7  relay that switches the SERVO POWER SUPPLY (HIGH = servo power on)
  Power the servos from their own supply (5-6 V, enough amps), common ground with the Arduino.
*/
#include <Servo.h>

const uint8_t NUM_JOINTS = 4;
const uint8_t JOINT_PINS[NUM_JOINTS] = {3, 5, 6, 9};
const uint8_t GRIPPER_PIN = 10;
const uint8_t ESTOP_PIN = 2;
const uint8_t RELAY_PIN = 7;
const uint8_t CURRENT_PIN = A0;

// Soft limits in servo degrees (match JOINT_LIMITS / SERVO_MAP on the host side)
const float SERVO_MIN[NUM_JOINTS] = {0, 0, 30, 0};
const float SERVO_MAX[NUM_JOINTS] = {180, 120, 170, 160};
// Park pose = servo angles for config.HOME_XYZ (computed with kinematics.to_servo). Recompute if you change
// the geometry or SERVO_MAP. NOTE: servos jump to this pose at power-up, so start with the arm near it.
const float HOME[NUM_JOINTS] = {180, 93, 73, 24};
const float GRIPPER_MIN = 15, GRIPPER_MAX = 125;

Servo joints[NUM_JOINTS];
Servo gripper;
float current[NUM_JOINTS];
float gripperDeg = 20;
bool stopped = false;

void cutPower() {
  digitalWrite(RELAY_PIN, LOW);
  for (uint8_t i = 0; i < NUM_JOINTS; i++) joints[i].detach();
  gripper.detach();
  stopped = true;
}

bool hardwareEstopPressed() { return digitalRead(ESTOP_PIN) == LOW; }

void enablePower() {
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    current[i] = HOME[i];
    joints[i].attach(JOINT_PINS[i]);
    joints[i].write((int)current[i]);
  }
  gripper.attach(GRIPPER_PIN);
  gripper.write((int)gripperDeg);
  digitalWrite(RELAY_PIN, HIGH);
  stopped = false;
}

void setup() {
  Serial.begin(115200);
  pinMode(ESTOP_PIN, INPUT_PULLUP);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  if (!hardwareEstopPressed()) enablePower(); else stopped = true;
}

// Move all joints together, each at a speed that makes them finish at the same time.
bool moveJoints(const float target[NUM_JOINTS], float dps) {
  float maxDelta = 0;
  float start[NUM_JOINTS];
  for (uint8_t i = 0; i < NUM_JOINTS; i++) {
    start[i] = current[i];
    float t = constrain(target[i], SERVO_MIN[i], SERVO_MAX[i]);
    maxDelta = max(maxDelta, fabs(t - start[i]));
  }
  if (maxDelta < 0.2) return true;
  unsigned long durationMs = (unsigned long)(maxDelta / dps * 1000.0);
  unsigned long t0 = millis();
  while (true) {
    if (hardwareEstopPressed()) { cutPower(); return false; }
    unsigned long dt = millis() - t0;
    float f = dt >= durationMs ? 1.0 : (float)dt / durationMs;
    for (uint8_t i = 0; i < NUM_JOINTS; i++) {
      float t = constrain(target[i], SERVO_MIN[i], SERVO_MAX[i]);
      current[i] = start[i] + (t - start[i]) * f;
      joints[i].write((int)(current[i] + 0.5));
    }
    if (f >= 1.0) break;
    delay(15);
  }
  return true;
}

void handleLine(String line) {
  line.trim();
  if (line.length() == 0) return;
  char cmd = line.charAt(0);

  if (cmd == 'P') { Serial.println("PONG"); return; }
  if (cmd == 'E') { cutPower(); Serial.println("OK"); return; }
  if (cmd == 'R') {
    if (hardwareEstopPressed()) { Serial.println("ESTOP"); return; }
    enablePower();
    Serial.println("OK");
    return;
  }
  if (cmd == 'Q') {
    Serial.print("S ");
    Serial.print(analogRead(CURRENT_PIN));
    Serial.print(" ");
    Serial.println((int)gripperDeg);
    return;
  }
  if (stopped || hardwareEstopPressed()) { cutPower(); Serial.println("ESTOP"); return; }

  if (cmd == 'J') {
    float v[NUM_JOINTS], dps;
    int n = sscanf(line.c_str() + 1, "%f %f %f %f %f", &v[0], &v[1], &v[2], &v[3], &dps);
    if (n != 5 || dps < 1) { Serial.println("ERR"); return; }
    Serial.println(moveJoints(v, dps) ? "OK" : "ESTOP");
    return;
  }
  if (cmd == 'G') {
    float deg = line.substring(1).toFloat();
    gripperDeg = constrain(deg, GRIPPER_MIN, GRIPPER_MAX);
    gripper.write((int)gripperDeg);
    Serial.println("OK");
    return;
  }
  Serial.println("ERR");
}

void loop() {
  if (hardwareEstopPressed() && !stopped) cutPower();
  if (Serial.available()) handleLine(Serial.readStringUntil('\n'));
}
