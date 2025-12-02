import cv2
import numpy as np

prototxt = "deploy.prototxt"
model = "mobilenet_iter_73000.caffemodel"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe(prototxt, model)
cap = cv2.VideoCapture(0)

person_detected_last = None
direction_last = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    person_detected = False
    direction = "Камера по центру"

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])
        if confidence > 0.5 and CLASSES[idx] == "person":
            person_detected = True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f"{CLASSES[idx]}: {confidence:.2f}"
            cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            frame_center = w // 2
            person_center = (startX + endX) // 2
            if person_center < frame_center - 30:
                direction = "Повернуть камеру влево"
            elif person_center > frame_center + 30:
                direction = "Повернуть камеру вправо"
            else:
                direction = "Камера по центру"
            break  

    
    if person_detected != person_detected_last:
        if person_detected:
            print("Человек обнаружен!")
        else:
            print("Человека нет.")
        person_detected_last = person_detected

    
    if direction != direction_last:
        print(direction)
        direction_last = direction

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
