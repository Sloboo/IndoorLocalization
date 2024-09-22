// LED and color pin mapping:
// LED3: {2, 3, 4}    - Pins 2 = RED, 3 = GREEN, 4 = BLUE
// LED2: {9, 10, 11}  - Pins 9 = RED, 10 = GREEN, 11 = BLUE
// LED1: {5, 6, 7}    - Pins 5 = RED, 6 = GREEN, 7 = BLUE
// LED4: {A5, A4, A3} - Pins A5 = RED, A4 = GREEN, A3 = BLUE
// LED5: {A2, A1, A0} - Pins A2 = RED, A1 = GREEN, A0 = BLUE

const int ledPins[][3] = {
  {5, 6, 7},    // LED1
  {9, 10, 11},  // LED2
  {2, 3, 4},    // LED3
  {A5, A4, A3}, // LED4
  {A2, A1, A0}  // LED5
};

void setup() {
  // Initialize all LED pins as outputs and set them to LOW initially
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 3; j++) {
      pinMode(ledPins[i][j], OUTPUT);
      digitalWrite(ledPins[i][j], LOW);
    }
  }
  Serial.begin(9600);
}

void loop() {
  // Group 1 arrays
  setLeds();
  analogWrite(5,255);
  analogWrite(9,25);
  analogWrite(3,25);
  digitalWrite(A5,HIGH);
  digitalWrite(A2,HIGH);
  delay(5000);

    // Group 1 arrays
  setLeds();
  analogWrite(6,255);
  analogWrite(11,25);
  analogWrite(3,25);
  digitalWrite(A4,HIGH);
  digitalWrite(A0,HIGH);
  delay(5000);

    // Group 1 arrays
  setLeds();
  digitalWrite(7,HIGH);
  analogWrite(10,25);
  analogWrite(3,25);
  digitalWrite(A3,HIGH);
  digitalWrite(A1,HIGH);
  delay(5000);
  
}




void setLeds() {

    // Set all LEDs to LOW
    for (int i = 0; i < 5; i++) {
      digitalWrite(ledPins[i][0], LOW); // Red off
      digitalWrite(ledPins[i][1], LOW); // Green off
      digitalWrite(ledPins[i][2], LOW); // Blue off
    }
}


