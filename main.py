from typing import Tuple
from models import dense_model, conv2d_model
from data_worker import get_data
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import numpy as np
import pandas as pd

tf.random.set_seed(1234)
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def fit_model(
    model: Model,
    checkpoint_path: str,
    train_data: Tuple[np.ndarray, np.ndarray],
    valid_data: Tuple[np.ndarray, np.ndarray],
    epochs: int = 10,
    batch_size: int = 32,
) -> None:
    """
    Function to train the passed model

    Args:
        model (Model): model to be trained
        checkpoint_path (str): file path to save the model
        train_data (Tuple[np.ndarray, np.ndarray]): training data
        valid_data (Tuple[np.ndarray, np.ndarray]): validation data
        epochs (int, optional): number of learning epochs. Defaults to 10.
        batch_size (int, optional): batch size. Defaults to 32.
    """
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=1e-4, verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)

    X_train = train_data[0]
    y_train = train_data[1]

    model.fit(
        X_train,
        y_train,
        validation_data=valid_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[model_checkpoint_callback, reduce_lr, early_stop],
    )
    
def predict_and_create_submission(checkpoint_path: str) -> None:
    """
    Function to predict the trained model 
    and save the result to the submission.csv file

    Args:
        checkpoint_path (str): the path to the checkpoint with the desired model
    """    
    loaded_model = load_model(checkpoint_path)
    preds = loaded_model.predict(scale_test_data)
    predictions = np.argmax(preds, axis=1)
    submission = pd.read_csv("digit-recognizer/sample_submission.csv")
    submission["Label"] = predictions
    submission.to_csv("submission.csv", index=False)

if __name__ == "__main__":
    # model = dense_model(input_shape=(784,), output_shape=10, units=[256,128,64,32, 16], lr = 1e-4) kaggle result =  # 0.96339
    # model = conv2d_model(input_shape=(28,28,1),
    #                      kernels=[[3,3],[3,3]],
    #                      filters=[150,300],
    #                      pools=[[2,2],[2,2]],
    #                      lr = 1e-3)  # kaggle result = 0.98821
    model = dense_model(
        input_shape=(784,), output_shape=10, units=[64, 32, 16], lr=1e-3
    )
    print(model.summary())
    X_train, X_val, y_train, y_val, scale_test_data = get_data(
        "digit-recognizer", conv_mod=False
    )
    checkpoint_path = "dense_mnist.h5"
    fit_model(model, checkpoint_path, (X_train, y_train), (X_val, y_val), epochs=150)
    predict_and_create_submission(checkpoint_path)
