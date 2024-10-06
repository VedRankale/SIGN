Changes:

Literally everything, rewrote the whole model, data capturing and displaying sections
Included on screen word display
Included voice output

Model: Model now runs at 100 epochs, reducing compile time. This means the model that used to take one
       hour to compile now takes 2 minutes
       Also switched to .keras instead of .h5 for better loading times

Data: Added better control for data capturing, which means you control when the sign is captured.
      Also reduced no. of frames for smoother sign conversion

Display: Added text and voice output
	 Reduced model size for smoother camera operation ( can now run on 500MB of ram)
	 Added text editing commands (press q to delete a word, space to speak and clear sentence etc.)

Libraries to install:
Mediapipe
scikit-learn
OpenCV 
TensorFlow
keyboard (pip install keyboard) [I will try to remove this library soon]
pyttsx3 (pip install pyttsx3)

If you have finished reading this, please contact me asap so I can explain in detail how to use this shiny
new program :]
