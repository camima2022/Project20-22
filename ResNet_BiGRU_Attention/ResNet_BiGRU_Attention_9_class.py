from __future__ import print_function,division
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv1D,MaxPooling1D
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import GRU,LSTM
from tensorflow.keras.optimizers import SGD

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2
from sklearn.metrics import f1_score, precision_score, recall_score
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Attention
import  DataCollection

length = 784
nb_classes = 9

path = '/home/hbb/pythonProject/ResNet_BiGRU_Attention/ISCX2012_2000_version/'
class_list = ['bittorrent','dns','ftp','http','imap','pop3','smb','smtp','ssh']
data, label = DataCollection.DataPayloadCollection_ISCX(path,class_list)

train_x, test_x , train_y,test_y = train_test_split(data, label, test_size=0.2, random_state=0, shuffle=True)
train_y = to_categorical(train_y, nb_classes)
test_y = to_categorical(test_y, nb_classes)
"""
train_x = train_x.reshape((train_x.shape[0], length, 1))
test_x = test_x.reshape((test_x.shape[0],length, 1))
"""
train_x = train_x.reshape((train_x.shape[0], 28,28, 1))
test_x = test_x.reshape((test_x.shape[0],28,28, 1))

class Metrics1(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        self.validation_data = (test_x, test_y)

        pred = self.model.predict(self.validation_data[0])
        val_predict = np.asarray(pred).round()
        val_targ = self.validation_data[1]

        _val_f1 = f1_score(val_targ, val_predict, average=None)
        _val_recall = recall_score(val_targ, val_predict, average=None)
        _val_precision = precision_score(val_targ, val_predict, average=None)

        val_precision = []
        val_recall = []
        val_f1 = []
        for i in range(len(_val_f1)):
            val_precision.append(round(_val_precision[i], nb_classes))
            val_recall.append(round(_val_recall[i], nb_classes))
            val_f1.append(round(_val_f1[i], nb_classes))

        print("\n")
        print('Eval      \t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format('0', '1', '2', '3','4', '5', '6', '7', '8'))
        print('Precision \t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(val_precision[0], val_precision[1], val_precision[2], val_precision[3],
                                                                      val_precision[4], val_precision[5], val_precision[6], val_precision[7], val_precision[8]))
        print('Recall    \t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(val_recall[0], val_recall[1],val_recall[2], val_recall[3],val_recall[4],
                                                                      val_recall[5],val_recall[6], val_recall[7],val_recall[7]))
        print("F1        \t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(val_f1[0], val_f1[1],val_f1[2], val_f1[3],val_f1[4], val_f1[5],val_f1[6], val_f1[7], val_f1[8]))
        self.val_f1s.append(val_f1)
        self.val_recalls.append(val_recall)
        self.val_precisions.append(val_precision)
Metrics = Metrics1()


def basic_block(x, input_filter):
    res_x = layers.BatchNormalization()(x)
    x1 = layers.Activation('relu')(res_x)
    x1 = layers.Conv1D(filters=input_filter, kernel_size=9, strides=1, padding='same')(x1)

    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv1D(filters=input_filter, kernel_size=9, strides=1, padding='same')(x1)

    x2 = layers.Activation('relu')(res_x)
    x2 = layers.Conv1D(filters=input_filter, kernel_size=5, strides=1, padding='same')(x2)

    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Activation('relu')(x2)
    x2 = layers.Conv1D(filters=input_filter, kernel_size=5, strides=1, padding='same')(x2)

    identity = layers.Conv1D(filters=input_filter, kernel_size=1, strides=1, padding="same")(x)

    output = layers.add([x1, x2 , identity])

    return output

def ResNet_Attention():
    input_shape = (length, 1)
    x_input = Input(input_shape)
    x = layers.Conv1D(8, 9, activation='relu', input_shape=(length, 1), padding='valid')(x_input)
    x = layers.MaxPooling1D(pool_size=3)(x)
    x = basic_block(x, 16)
    x = basic_block(x, 16)
    x = layers.MaxPooling1D(pool_size=3)(x)
    h = layers.Bidirectional(LSTM(10, return_sequences=True))(x)
    atten_out = Attention()([h, h])
    x = layers.Flatten()(atten_out)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(nb_classes, activation='softmax')(x)

    model = Model(inputs=x_input, outputs=output)

    model.summary()
    return model

def PrtCNN():
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format='channels_last',
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last'))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding="same", data_format='channels_last',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last'))
    #model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format='channels_last',
    #               activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=nb_classes, activation='softmax'))  # kernel_regularizer=l2(0.01)
    model.summary()
    return model

def MMN_CNN():
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format='channels_last',
                     activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last'))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding="same", data_format='channels_last',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last'))
    model.add(Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), padding="same", data_format='channels_last',
                     activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=nb_classes, activation='softmax', kernel_regularizer=l2(0.01)))
    model.summary()
    return model

#model = ResNet_Attention()
#model = PrtCNN()
model = PrtCNN()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer= sgd,loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=40, batch_size=128, verbose=2,validation_data=(test_x, test_y),callbacks=[Metrics])
score = model.evaluate(test_x, test_y, verbose=1)
print('TestSet loss:', score[0])
print('TestSet accuracy:', score[1])

val_accuracy = history.history['val_accuracy']
for i in range(len(val_accuracy)):
     val_accuracy[i] = round(val_accuracy[i], nb_classes)


val_accuracy = history.history['val_accuracy']
acc = history.history['accuracy']
test_loss = history.history['val_loss']
train_loss = history.history['loss']

for i in range(len(val_accuracy)):
    val_accuracy[i] = round(val_accuracy[i], nb_classes)
    acc[i] = round(acc[i], nb_classes)
    test_loss[i] = round(test_loss[i], nb_classes)
    train_loss[i] = round(train_loss[i], nb_classes)
print(val_accuracy)
print(acc)
print(test_loss)
print(train_loss)