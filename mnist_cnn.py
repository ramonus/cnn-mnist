#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, LeakyReLU, Flatten
from keras.datasets import mnist
from keras.callbacks import Callback
import keras
from pathlib import Path
    
def conv(input_shape=(28,28,1), dr=0.2):
    """
    Creates convolution model for recognition
    :param input_shape: Shape of the image input
    :param dr: Dropout rate
    :return model: Model built but not compiled
    """

    model = Sequential()
    depth = 32
    model.add(Conv2D(depth, kernel_size=(3,3), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(depth*2, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(dr))
    model.add(Conv2D(depth*4, kernel_size=(2,2), activation='relu'))
    model.add(Conv2D(depth*8, kernel_size=(2,2), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10,activation='softmax'))
    return model

class Hist(Callback):
    def on_train_begin(self,logs={}):
        self.acc = []
        self.loss = []
    def on_batch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))


def conv_nn():
    p = Path("mnist_cnn.h5")
    if p.exists():
        c = load_model(str(p))
    else:
        c = conv()
    c.summary()
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    
    optimizer = keras.optimizers.Adadelta()
    x_train = x_train.reshape(60000,28,28,1)
    x_test = x_test.reshape(10000,28,28,1)
    y_train = keras.utils.to_categorical(y_train,10)
    y_test = keras.utils.to_categorical(y_test, 10)
    c.compile(loss=keras.losses.binary_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    epochs = 10
    batch_size = 100
    hist = Hist()
    c.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[hist])
    score = c.evaluate(x_test,y_test,batch_size=batch_size)
    print("Score:",score)

    c.save("mnist_cnn.h5")

    f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    
    ax1.plot(hist.acc,'b',label='acc')
    ax1.set_title("Accuracy")
    ax1.grid()
    ax1.legend()

    ax2.plot(hist.loss,'r',label='loss')
    ax2.set_title("Loss")
    ax2.grid()
    ax2.legend()

    ax3.plot(hist.acc,'b',label="acc")
    ax3.plot(hist.loss,'r',label='loss')
    ax3.grid()
    ax3.legend()

    plt.show()
def fcnn():
    m1 = fully_connected_relu()
    m2 = fully_connected_LeReLU()
    opt = keras.optimizers.RMSprop(lr=0.1)
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    
    y_train = keras.utils.to_categorical(y_train,10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    m1.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])
    m2.compile(loss=keras.losses.binary_crossentropy, optimizer=opt, metrics=['accuracy'])

    epochs = 1
    batch_size = 100
    h1 = Hist()
    h2 = Hist()
    m1.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[h1])
    m2.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[h2])
    
    f, (ax1, ax2) = plt.subplots(1,2)
    f.suptitle("Accuracy and loss logs")
    ax1.set_title("FC_ReLU")
    ax1.plot(h1.acc,'b',label='acc')
    ax1.plot(h1.loss,'r',label='loss')
    ax1.grid()
    ax1.legend()
    ax2.set_title("FC_LeReLU")
    ax2.plot(h2.acc, 'b', label='acc')
    ax2.plot(h2.loss, 'r', label='loss')
    ax2.grid()
    ax2.legend()
    plt.show()

if __name__=="__main__":
    conv_nn()