import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.models import model_from_json
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import json
import os

X = np.asarray(pickle.load(open("X.pickle", "rb")))
y = np.asarray(pickle.load(open("y.pickle", "rb")))

X = X/255.0

NETWORK_PARAMETERS_DIR = 'network_parameters'
MODELS_DIR = 'models'
network_parameter_files = os.listdir(NETWORK_PARAMETERS_DIR)
network_parameter_files.sort()

file_index = 0
while file_index < 10:
    file = network_parameter_files[file_index]
    print("Generating model from " + file)
    json_file = open(os.path.join(NETWORK_PARAMETERS_DIR,file))
    json_string = json_file.read()
    json_data = json.loads(json_string)
    activation_function = json_data["activation_function"]
    batch_size = json_data["batch_size"]
    epochs = json_data["epoch"]
    learning_rate = json_data["learning_rate"]
    hidden_layer_nodes = json_data["hidden_layer_nodes"]

    model = Sequential()
    # 3 convolutional layers
    model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # 2 hidden layers
    model.add(Flatten())
    model.add(Dense(hidden_layer_nodes))
    model.add(Activation("relu"))

    model.add(Dense(hidden_layer_nodes))
    model.add(Activation("relu"))

    # The output layer with 6 neurons, for 6 classes
    model.add(Dense(6))
    model.add(Activation("softmax"))

    adam_optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss="sparse_categorical_crossentropy",
    				optimizer=adam_optimizer,
    				metrics=["accuracy"])


    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    model_json = model.to_json()
    #Save model to Models_Folder/Model_Folder/..
    #np_0_model
    model_folder = file.rsplit('.', 1)[0] + "model"
    #models/np_0_model
    model_folder = os.path.join(MODELS_DIR, model_folder)
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    #models/np_0_model/model.json
    with open(os.path.join(model_folder, "model.json"), "w") as json_file :
    	json_file.write(model_json)

    #models/np_0_model/model.h5

    model.save_weights(os.path.join(model_folder,"model.h5"))
    print("Saved model to disk")

    #models/np_0_model/CNN.model
    model.save(os.path.join(model_folder,'CNN.model'))
    file_index += 1
