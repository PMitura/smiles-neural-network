import data
from keras.models import Sequential
from keras.layers import Activation, Dense, LSTM

# RNN parameters
MAX_BACKWARD_STEPS = 100

# 3D input: dimensions are (number of samples, length of word, alphabet)
# TODO: process 1 of k encoding
def setup(alphaSize):
    print('  Compiling model...')

    model = Sequential()
    model.add(LSTM(32, activation = 'sigmoid', input_shape =
        (min(alphaSize, MAX_BACKWARD_STEPS), alphaSize)))
    model.add(Dense(1))
    model.compile(loss = 'mse', optimizer = 'rmsprop')

    print('  ...done')
    return model


def train(model, nnInput, refOutput):
    print('  Training model...')
    # model.fit(nnInput, refOutput, nb_epoch = 5, batch_size = 16)
    # TODO: repair
    print('  ...done')


def test(model, nnInput, refOutput):
    # score = model.evaluate(nnInput, refOutput, batch_size = 16)
    # print("Score on traning data: {}".format(score))
    # TODO: repair
    return


def main():
    nnInput, refOutput, alphaSize = data.prepareData()
    model = setup(alphaSize)
    train(model, nnInput, refOutput)
    test(model, nnInput, refOutput)


if __name__ == '__main__':
    main()
