#All the libraries and functions we need

import mediapipe as pipe #Mediapipe to draw landmarks
import numpy as num #Numpy to format and save data
import cv2 as cam #Camera library
import os #Computer functions [need to change this Ved]
import time #Not really important can remove

#List of all Tensorflow Models

import tensorflow.keras.models as ld
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM,Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split #SKLearn model to configure data

data_path = os.path.join('sign_data') #Our data path folder
mp_holistic = pipe.solutions.holistic #Holistic input model
mp_drawing = pipe.solutions.drawing_utils #Holistic drawing model
seq_num = 30    #Number of sequences
seq_len = 30    #Number of frames
start_folder = 30 #Number of folders (same as frames)
actions = num.array(['namaste','yo','thank you'])    #The words for signing
label_map = {label:num for num, label in enumerate(actions)}    #Numbering the words
log_dir = os.path.join('Logs')  #Folder for model training data
tb_callback = TensorBoard(log_dir=log_dir)  #Idrk what this is tbh
colors = [(245,117,16), (117,245,16), (16,117,245)] #Walke chose these colours

def mp_detect(image, model):    #Converting the image in terms the model can read
    image = cam.cvtColor(image, cam.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cam.cvtColor(image, cam.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results): #Draws normal dots and lines on live feed
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):  #Draws kewl dots and lines on live feed
     mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results): #Takes all the numpy data from mediapipe keypoints
    pose = num.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else num.zeros(33*4)
    face = num.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else num.zeros(468*3)
    lh = num.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else num.zeros(21*3)
    rh = num.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else num.zeros(21*3)
    return num.concatenate([pose, face, lh, rh])


def prob_viz(res, actions, input_frame, colors): #So this compiles all the output data for final showing
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cam.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cam.putText(output_frame, actions[num], (0, 85 + num * 40), cam.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cam.LINE_AA)

    return output_frame

