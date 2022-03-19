from typing import Tuple, List
from tensorflow.keras.layers import Conv2D, Dense, Input, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def dense_model(
    input_shape: Tuple[int, ...], 
    output_shape: int,
    units: List[int],
    lr: float
) -> Model:
    """
    Function for building a fully connected model

    Args:
        input_shape (Tuple[int,...]): input resolution
        output_shape (int): output resolution
        units (List[int]):  dimensionality of the output space for each dense layer
        lr (float): learning rate

    Returns:
        Model: dense model
    """
    input_layer = Input(input_shape)
    d = Dense(units[0], activation="relu")(input_layer)

    for unit in units[1:]:
        d = Dense(unit, activation="relu")(d)

    output_layer = Dense(output_shape, activation="softmax")(d)
    model = Model(inputs=[input_layer], outputs=[output_layer])
    model.compile(
        optimizer=Adam(lr=lr), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def conv2d_model(
    input_shape: Tuple[int, ...],
    kernels: List[int],
    filters: List[int],
    pools: List[int],
    lr: float,
) -> Model:
    """
    Function for building a convolution model

    Args:
        input_shape (Tuple[int, ...]): input resolution
        kernels (List[int]): convolution kernel size for each layer
        filters (List[int]): number of filters for each layer
        pools (List[int]): pooling window size
        lr (float): learning rate

    Returns:
        Model: convolution model
    """
    input_layer = Input(input_shape)
    conv2d = Conv2D(filters=filters[0], kernel_size=kernels[0], activation="relu")(
        input_layer
    )
    maxpool = MaxPooling2D(pool_size=pools[0])(conv2d)

    for i in range(len(kernels[1:])):
        conv2d = Conv2D(filters=filters[i], kernel_size=kernels[i], activation="relu")(maxpool)
        maxpool = MaxPooling2D(pool_size=pools[0])(conv2d)

    flatten = Flatten()(conv2d)
    output_dense = Dense(10, activation="softmax")(flatten)
    model = Model(inputs=[input_layer], outputs=[output_dense])
    model.compile(
        optimizer=Adam(lr=lr), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
