from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from functions import image_process, keypoint_extraction, speak
import mediapipe as mp

app = Flask(__name__)

# Load the trained model and other configurations from main.py
PATH = os.path.join('sign_data')
actions = np.array(os.listdir(PATH))
model = load_model('my_model.keras')

cap = cv2.VideoCapture(0)

def generate_frames():
    sentence, keypoints, last_prediction = [], [], []
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
                results = image_process(frame, holistic)
                keypoints.append(keypoint_extraction(results))
                if len(keypoints) == 10:
                    keypoints = np.array(keypoints)
                    prediction = model.predict(keypoints[np.newaxis, :, :], verbose=0)
                    keypoints = []
                    if np.amax(prediction) > 0.9 and last_prediction != actions[np.argmax(prediction)]:
                        sentence.append(actions[np.argmax(prediction)])
                        last_prediction = actions[np.argmax(prediction)]
                if len(sentence) > 5:
                    sentence = sentence[-5:]
                if sentence:
                    cv2.putText(frame, ' '.join(sentence), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
