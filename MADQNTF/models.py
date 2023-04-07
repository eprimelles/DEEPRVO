from keras.models import Model
from keras.layers import Dense, Input, Reshape


def Q_Net (input_dim, output_dim):

    inputs = Input(input_dim)

    h1 = Dense(128, activation='relu')(inputs)
    h2 = Dense(64, activation='relu')(h1)
    h3 = Dense(32, activation='relu')(h2)

    outputs = Dense(output_dim, activation='softmax')(h3)
    

    return Model(inputs, outputs)


if __name__ == '__main__':

    import numpy as np
    x = Q_Net(8, 8)
    data = np.random.random((1, 8))
    print( np.squeeze(np.argmax(x(data), 1)))