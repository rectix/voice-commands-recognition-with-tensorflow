import numpy as np

from tensorflow import keras

from recording  import record_audio, terminate
from tensor import preprocess_audiobuffer

# !! Modify this in the correct order
commands = ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']



loaded_model = keras.models.load_model("saved_model")

def predict_mic():
    audio = record_audio()
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    label_pred = np.argmax(prediction, axis=1)
    command = commands[label_pred[0]]
    print("Predicted label:", command)
    return command

if __name__ == "__main__":
    from turtle_file import move_turtle
    while True:
        command = predict_mic()
        move_turtle(command)
        if command == "stop":
            terminate()
            break
 

 


