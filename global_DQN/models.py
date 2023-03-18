from keras.models import Model
from keras.layers import Dense, Input, Reshape


def Q_Net (input_dim, output_dim):

    inputs = Input(input_dim)

    h1 = Dense(128, activation='relu')(inputs)
    h2 = Dense(64, activation='relu')(h1)
    h3 = Dense(32, activation='relu')(h2)

    outputs = Dense(output_dim[0] * output_dim[1], activation='softmax')(h3)
    outputs = Reshape(output_dim)(outputs)

    return Model(inputs, outputs)