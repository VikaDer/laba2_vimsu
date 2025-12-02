import cv2
import numpy as np
import RPi.GPIO as GPIO
import time

# --- Настройка GPIO
BUTTON_PIN = 5
LED_PIN = 23

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

prototxt = "deploy.prototxt"
model = "mobilenet_iter_73000.caffemodel"
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe(prototxt, model)
cap = cv2.VideoCapture(0)

try:
    while True:
        if GPIO.input(BUTTON_PIN) == GPIO.HIGH:
            ret, frame = cap.read()
            if not ret:
                print("Can't find the frame")
                continue

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()

            person_detected = False

            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                idx = int(detections[0, 0, i, 1])
                if confidence > 0.5 and CLASSES[idx] == "person":
                    person_detected = True
                    break

            if person_detected:
                GPIO.output(LED_PIN, GPIO.HIGH)
                print("Human detected! LED ON")
            else:
                GPIO.output(LED_PIN, GPIO.LOW)
                print("Human not found. LED OFF")

            time.sleep(0.1)
        else:
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(0.1)

except KeyboardInterrupt:
    print("Program finished")
finally:
    GPIO.output(LED_PIN, GPIO.LOW)
    GPIO.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    print("GPIO cleared, camera closed")
