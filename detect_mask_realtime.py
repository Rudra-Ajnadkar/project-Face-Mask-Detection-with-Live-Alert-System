import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model/mask_detector_model.h5')
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(frame, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        resized = cv2.resize(face, (100, 100)) / 255.0
        reshaped = np.reshape(resized, (1, 100, 100, 3))
        result = model.predict(reshaped)[0][0]

        label = "Mask" if result > 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if label == "No Mask":
            print("⚠️ ALERT: No Mask Detected!")

    cv2.imshow("Face Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
