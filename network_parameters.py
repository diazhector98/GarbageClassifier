import json
import os

activation_functions = ["relu", "tanh"]
batch_sizes = [16,32]
epochs = [80,160]
learning_rates = [0.0005,0.001,0.0015]
hidden_layer_nodes = [128,256]

DIR = 'network_parameters'
index = 0
for activation_function in activation_functions:
    for batch_size in batch_sizes:
        for epoch in epochs:
            for learning_rate in learning_rates:
                for hidden_layer_node in hidden_layer_nodes:
                    parameters = {
                        "activation_function": activation_function,
                        "batch_size": batch_size,
                        "epoch": epoch,
                        "learning_rate": learning_rate,
                        "hidden_layer_nodes": hidden_layer_node
                    }
                    file = os.path.join(DIR, 'np_' + str(index) + "_.txt" )
                    index += 1
                    with open(file, 'w') as outfile:
                        json.dump(parameters, outfile)
