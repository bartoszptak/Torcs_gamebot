import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Concatenate, Dense, CuDNNLSTM

def get_model():
    speed = Input(shape=(4,2), name='speed_input')
    img = Input(shape=(64,64,12), name='img_input')

    conv_0 = Conv2D(32, 3, strides=(1, 1), padding='same',
                        activation='relu')(img)
    conv_1 = Conv2D(64, 3, strides=(2, 2), padding='same',
                        activation='relu')(conv_0)

    maxpool_1 = MaxPool2D(padding='same')(conv_1)
    conv_2 = Conv2D(64, 3, padding='same', activation='relu')(maxpool_1)

    maxpool_2 = MaxPool2D(padding='same')(conv_2)
    conv_3 = Conv2D(128, 3, padding='same', activation='relu')(maxpool_2)

    flatten_0 = Flatten()(conv_3)
    fc_0 = Dense(64)(flatten_0)
    fc_1 = CuDNNLSTM(64)(speed)

    con_0 = Concatenate()([fc_0, fc_1])

    steer = Dense(1, activation='tanh', name='steer_output')(con_0)
    acc = Dense(1, activation='tanh', name='acc_output')(con_0)

    model = tf.keras.models.Model([img, speed], [steer, acc])
    model.compile(optimizer='adam', loss='mse')

    return model