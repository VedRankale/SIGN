
import numpy as np
import os
import string
import mediapipe as mp
import cv2
from functions import *
import keyboard
from tensorflow.keras.models import load_model

# Set the path to the data directory
PATH = os.path.join('sign_data')

# Create an array of action labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Load the trained model
model = load_model('my_model.keras')

# Initialize the lists
sentence, keypoints, last_prediction= [], [], []

# Access the camera and check if the camera is opened successfully


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a holistic object for sign prediction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Run the loop while the camera is open
    while cap.isOpened():
        # Read a frame from the camera
        _, image = cap.read()
        # Process the image and obtain sign landmarks using image_process function from my_functions.py
        results = image_process(image, holistic)
        image = cv2.flip(image, 1)
        # Extract keypoints from the pose landmarks using keypoint_extraction function from my_functions.py
        keypoints.append(keypoint_extraction(results))

        # Check if 10 frames have been accumulated
        if len(keypoints) == 10:
            # Convert keypoints list to a numpy array
            keypoints = np.array(keypoints)
            # Make a prediction on the keypoints using the loaded model
            prediction = model.predict(keypoints[np.newaxis, :, :],verbose=0)
            # Clear the keypoints list for the next set of frames
            keypoints = []

            # Check if the maximum prediction value is above 0.9
            if np.amax(prediction) > 0.9:
                # Check if the predicted sign is different from the previously predicted sign
                if last_prediction != actions[np.argmax(prediction)]:
                    # Append the predicted sign to the sentence list
                    sentence.append(actions[np.argmax(prediction)])
                    # Record a new prediction to use it on the next cycle
                    last_prediction = actions[np.argmax(prediction)]

        # Limit the sentence length to 7 elements to make sure it fits on the screen
        if len(sentence) > 5:
            sentence = sentence[-5:]

        # Reset if the "Spacebar" is pressed
        if keyboard.is_pressed(' '):
            speak(sentence)
            sentence, keypoints, last_prediction = [], [], []

        if keyboard.is_pressed('q'):
            sentence = sentence[:-1]

        # Capitalize the first word of the sentence
        if sentence:
            sentence[0] = sentence[0].capitalize()

        # Check if the sentence has at least two elements
        if len(sentence) >= 2:
            # Check if the last element of the sentence belongs to the alphabet
            if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
                # Check if the second last element of sentence is valid
                if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                    # Combine last two elements
                    sentence[-1] = sentence[-2] + sentence[-1]
                    sentence.pop(len(sentence) - 2)
                    sentence[-1] = sentence[-1].capitalize()

        if sentence:
            # Join the sentence list into a string and display it
            text = ' '.join(sentence)
            cv2.putText(image,text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

        # Show the image on the display
        cv2.imshow('Camera', image)

        cv2.waitKey(1)

        # Check if the 'Camera' window was closed and break the loop
        if keyboard.is_pressed('esc'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
