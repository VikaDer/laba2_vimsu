import cv2
import numpy as np
from gpiozero import Button, LED
from time import sleep

# --- Настройка GPIO
BUTTON_PIN = 5
LED_PIN = 23

# Используем внутренний подтягивающий резистор (pull_up=True по умолчанию для Button)
button = Button(BUTTON_PIN, pull_up=True)  # pull_up=True соответствует GPIO.PUD_UP
led = LED(LED_PIN)
led.off()

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
        # В GPIO Zero состояние кнопки проверяется через свойство is_pressed
        # Когда кнопка не нажата (подтяжка к HIGH), is_pressed = False
        # Когда кнопка нажата (контакт с GND), is_pressed = True
        # В исходном коде: GPIO.HIGH = кнопка не нажата, GPIO.LOW = нажата
        
        if button.is_pressed:  # Эквивалентно GPIO.input(BUTTON_PIN) == GPIO.HIGH
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
                led.on()  # Эквивалентно GPIO.output(LED_PIN, GPIO.HIGH)
                print("Human detected! LED ON")
            else:
                led.off()  # Эквивалентно GPIO.output(LED_PIN, GPIO.LOW)
                print("Human not found. LED OFF")

            sleep(0.1)
        else:
            led.off()
            sleep(0.1)

except KeyboardInterrupt:
    print("Program finished")
finally:
    led.off()  # Гарантированно выключаем LED
    # В GPIO Zero cleanup происходит автоматически при завершении программы
    cap.release()
    cv2.destroyAllWindows()
    print("GPIO cleared, camera closed")
