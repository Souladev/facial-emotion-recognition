import cv2
import numpy as np
from keras.models import load_model

# Load the model
model = load_model('model_file.h5')

face_cascade_path = 'C:/Users/tlili/OneDrive/Bureau/facial emotion recognition/data/haarcascade_frontalface_default.xml'
faceDetect = cv2.CascadeClassifier(face_cascade_path)

# Check if cascade file is loaded properly
if faceDetect.empty():
    print("Error loading face cascade file. Check the path!")
    exit()

# Define emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load image
frame = cv2.imread('C:/Users/tlili/OneDrive/Bureau/facial emotion recognition/data/ks.png')

# Check if image is loaded
if frame is None:
    print("Error loading image. Check the file path!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

# Check if faces are detected
if len(faces) == 0:
    print("No faces detected!")
else:
    for x, y, w, h in faces:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))

        # Make prediction
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        print(f"Detected Emotion: {labels_dict[label]}")

        # Draw rectangles
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Display the result
cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
