import pickle
from typing import Dict, List, Any, Union
import numpy as np
# Keras
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.utils import pad_sequences



def load_data() -> Dict[str, Union[List[Any], int]]:
    path = "/Users/jonasolsen/Documents/Skole/IIkt/4_semester/TDT4171-Metoder_i_kunstig_intelligens/Øvinger/øving_8/MetoderTiki8/keras-data.pickle"
    with open(file=path, mode="rb") as file:
        data = pickle.load(file)

    return data


def preprocess_data(data: Dict[str, Union[List[Any], int]]) -> Dict[str, Union[List[Any], np.ndarray, int]]:
    """
    Preprocesses the data dictionary. Both the training-data and the test-data must be padded
    to the same length; play around with the maxlen parameter to trade off speed and accuracy.
    """
    maxlen = data["max_length"]//16
    data["x_train"] = pad_sequences(data['x_train'], maxlen=maxlen)
    data["y_train"] = np.asarray(data['y_train'])
    data["x_test"] = pad_sequences(data['x_test'], maxlen=maxlen)
    data["y_test"] = np.asarray(data['y_test'])

    return data


def train_model(data: Dict[str, Union[List[Any], np.ndarray, int]], model_type="feedforward") -> float:
    """
    Build a neural network of type model_type and train the model on the data.
    Evaluate the accuracy of the model on test data.

    :param data: The dataset dictionary to train neural network on
    :param model_type: The model to be trained, either "feedforward" for feedforward network
                        or "recurrent" for recurrent network
    :return: The accuracy of the model on test data
    """

    # TODO build the model given model_type, train it on (data["x_train"], data["y_train"])
    #  and evaluate its accuracy on (data["x_test"], data["y_test"]). Return the accuracy
    print(" ------------------------------------------DATA-------------------------------------------")
    print(data)
    print(" -----------------------------------------------------------------------------------------")
    print(" ------------------------------------------SHAPES-----------------------------------------")
    print("x train shape: ", data["x_train"].shape)
    x_train_shape = data["x_train"].shape
    print("y train shape: ", data["y_train"].shape)
    y_train_shape = data["y_train"].shape
    print("x test shape: ", data["x_test"].shape)
    x_test_shape = data["x_test"].shape
    print("y test shape: ", data["y_test"].shape)  
    y_test_shape = data["y_test"].shape  
    print("vocabulary size: ", data["vocab_size"])
    vocab_size = data["vocab_size"]
    print("max length: ", data["max_length"])
    max_length = data["max_length"]
    print(" -----------------------------------------------------------------------------------------")


    x_train = tf.keras.utils.pad_sequences(data["x_train"], maxlen=max_length)
    y_train = tf.keras.utils.pad_sequences(data["y_train"].reshape(393053,1), maxlen=max_length)
    x_test = tf.keras.utils.pad_sequences(data["x_test"], maxlen=max_length)
    y_test = tf.keras.utils.pad_sequences(data["y_test"].reshape(130528,1), maxlen=max_length)

    
    if model_type=="feedforward": 
        model = tf.keras.Sequential(
            [
            layers.Embedding(input_dim=vocab_size, output_dim=1, input_length=max_length),
            layers.Dense(1, "relu")
            ]
        )

    
        print(model.summary())
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    metrics=tf.keras.metrics.BinaryAccuracy())
        history = model.fit(x_train, y_train, batch_size=32)
        results = model.evaluate(x_test, y_test, batch_size=128)
        predictions = model.predict(x_test[:3])

        print(results)

        return results[1]
    
    pass






def main() -> None:
    print("1. Loading data...")
    keras_data = load_data()
    print("2. Preprocessing data...")
    keras_data = preprocess_data(keras_data)
    print("3. Training feedforward neural network...")
    fnn_test_accuracy = train_model(keras_data, model_type="feedforward")
    print('Model: Feedforward NN.')
    print(f'Test accuracy: {fnn_test_accuracy:.3f}')
    print("4. Training recurrent neural network...")
    rnn_test_accuracy = train_model(keras_data, model_type="recurrent")
    print('Model: Recurrent NN.\n'
          f'Test accuracy: {rnn_test_accuracy:.3f}')



if __name__ == '__main__':
    main()

