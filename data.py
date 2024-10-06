import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
import keyboard
from functions import *

actions = np.array(['come'])

sequences = 30
frames = 10

PATH = os.path.join('sign_data')

for action, sequence in product(actions, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass

cap = cv2.VideoCapture(0)

with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    while cap.isOpened():
        for action, sequence, frame in product(actions, range(sequences), range(frames)):
            if frame == 0:
                while True:
                    if keyboard.is_pressed(' '):
                        break
                    _, image = cap.read()

                    # Flip the image horizontally for display
                    display_image = image

                    # Use the original image for processing
                    results = image_process(image, holistic)
                    draw_landmarks(display_image, results)
                    display_image = cv2.flip(display_image,1)

                    cv2.putText(display_image, 'Recording data for the "{}". Sequence number {}.'.format(action, sequence),
                                (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.putText(display_image, 'Pause.', (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(display_image, 'Press "Space" when you are ready.', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Camera', display_image)
                    cv2.waitKey(1)

                    if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                        break
            else:
                _, image = cap.read()

                # Flip the image horizontally for display
                display_image = image

                # Use the original image for processing
                results = image_process(image, holistic)
                draw_landmarks(display_image, results)
                display_image = cv2.flip(display_image,1)

                cv2.putText(display_image, 'Recording data for the "{}". Sequence number {}.'.format(action, sequence),
                            (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('Camera', display_image)
                cv2.waitKey(1)

            if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                break

            # Extract the landmarks using the original image for accuracy
            keypoints = keypoint_extraction(results)
            frame_path = os.path.join(PATH, action, str(sequence), str(frame))
            np.save(frame_path, keypoints)

        cap.release()
        cv2.destroyAllWindows()

