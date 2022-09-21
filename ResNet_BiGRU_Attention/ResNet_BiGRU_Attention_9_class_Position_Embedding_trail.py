from __future__ import print_function,division
import math
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras import layers

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import GRU,LSTM
from tensorflow.keras.optimizers import SGD,Adam

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

path = '/ResNet_BiGRU_Attention/raw_ISCX2012/'
class_list = ['bittorrent','dns','ftp','http','imap','pop3','smb','smtp','ssh']
data, label = DataCollection.DataPayloadCollection_ISCX(path,class_list)

train_x, test_x , train_y,test_y = train_test_split(data, label, test_size=0.2, random_state=0, shuffle=True)
train_y = to_categorical(train_y, nb_classes)
test_y = to_categorical(test_y, nb_classes)

train_x = train_x.reshape((train_x.shape[0], length, 1))
test_x = test_x.reshape((test_x.shape[0],length, 1))

"""
#2D CNN
train_x = train_x.reshape((train_x.shape[0], 28,28, 1))
test_x = test_x.reshape((test_x.shape[0],28,28, 1))
"""
def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
  """
  Gets a bunch of sinusoids of different frequencies.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position

  Returns:
    a Tensor of timing signals [1, length, channels]
  """
  #position = tf.cast(tf.range(length) + start_index,dtype = tf.float32)
  position = tf.cast(tf.range(length) + start_index,dtype = tf.float32)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.cast(num_timescales,dtype=tf.float32) - 1))
  inv_timescales = min_timescale * tf.exp(
      tf.cast(tf.range(num_timescales),dtype = tf.float32) * -log_timescale_increment)
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  signal = tf.pad(signal, [[0, 0], [0, tf.math.floormod(channels, 2)]])
  signal = tf.reshape(signal, [1, length, channels])
  return signal
def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
  """Adds a bunch of sinusoids of different frequencies to a Tensor.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  experessed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    start_index: index of first position

  Returns:
    a Tensor the same shape as x.
  """
  length = x.shape[1]
  channels = x.shape[2]
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale,
                                start_index)
  return x + signal


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
    print('Eval      \t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format('0', '1', '2', '3', '4', '5', '6', '7', '8'))
    print('Precision \t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(val_precision[0], val_precision[1], val_precision[2],
                                                                  val_precision[3],
                                                                  val_precision[4], val_precision[5], val_precision[6],
                                                                  val_precision[7], val_precision[8]))
    print('Recall    \t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(val_recall[0], val_recall[1], val_recall[2],
                                                                  val_recall[3], val_recall[4],
                                                                  val_recall[5], val_recall[6], val_recall[7],
                                                                  val_recall[7]))
    print("F1        \t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(val_f1[0], val_f1[1], val_f1[2], val_f1[3], val_f1[4],
                                                                  val_f1[5], val_f1[6], val_f1[7], val_f1[8]))
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

  output = layers.add([x1, x2, identity])

  return output
def ResNet_Attention():
  input_shape = (length,1)
  x_input = Input(input_shape)
  #embedded = layers.Embedding(input_dim=1000, output_dim = 128,input_length=784)(x_input)
  #position_embedded = add_timing_signal_1d(x = word_embedded)
  x = layers.Conv1D(16, 9, activation='relu', input_shape=(length, 128), padding='valid')(x_input)
  x = layers.MaxPooling1D(pool_size=5)(x)
  x = basic_block(x, 32)
  #x = basic_block(x, 16)
  x = layers.MaxPooling1D(pool_size=5)(x)
  h = layers.Bidirectional(GRU(10, return_sequences=True))(x)
  atten_out = Attention()([h, h])
  x = layers.Flatten()(atten_out)
  x = layers.Dropout(0.5)(x)
  output = layers.Dense(nb_classes, activation='softmax')(x)

  model = Model(inputs=x_input, outputs=output)

  model.summary()
  return model
def PrtCNN():
  model = tf.keras.models.Sequential()
  model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format='channels_last',
                   activation='relu', input_shape=(28, 28, 1)))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last'))
  model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding="same", data_format='channels_last',
                   activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last'))
  # model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format='channels_last',
  #               activation='relu'))
  # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last'))
  model.add(layers.Flatten())
  model.add(layers.Dense(units=128, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(units=nb_classes, activation='softmax'))  # kernel_regularizer=l2(0.01)
  model.summary()
  return model
def LiDaoQuan():
  model = tf.keras.models.Sequential()
  model.add(layers.Conv1D(filters=16, kernel_size=5, strides=1, padding='same', input_shape=(length, 1), activation='relu'))
  model.add(layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
  model.add(layers.AveragePooling1D(pool_size=2, strides=2, padding='same'))
  model.add(layers.Flatten())
  model.add(layers.Dense(60, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(50, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(40, activation='relu'))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(nb_classes, activation='softmax'))
  model.summary()
  return model
def MMN_CNN():
  model = tf.keras.models.Sequential()
  model.add(layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="same", data_format='channels_last',
                   activation='relu', input_shape=(28, 28, 1)))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last'))
  model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding="same", data_format='channels_last',
                   activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', data_format='channels_last'))
  model.add(layers.Conv2D(filters=128, kernel_size=(4, 4), strides=(1, 1), padding="same", data_format='channels_last',
                   activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(units=nb_classes, activation='softmax', kernel_regularizer=l2(0.01)))
  model.summary()
  return model
model = ResNet_Attention()
#model = PrtCNN()
#model = MMN_CNN()

#model = LiDaoQuan()

adam = Adam(learning_rate=0.001,decay=1e-6)
model.compile(optimizer= adam,loss = 'categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=20, batch_size=128, verbose=2,validation_data=(test_x, test_y))
score = model.evaluate(test_x, test_y, verbose=1)
print('TestSet loss:', score[0])
print('TestSet accuracy:', score[1])


val_accuracy = history.history['val_accuracy']
for i in range(len(val_accuracy)):
     val_accuracy[i] = round(val_accuracy[i], nb_classes)


val_accuracy = history.history['val_accuracy']
acc = history.history['accuracy']
test_loss = history.history['loss']
train_loss = history.history['val_loss']

"""
input = layers.Input(shape = (784,),dtype=tf.float32)

embedded = layers.Embedding(input_shape=(None,),input_dim=10,output_dim=256,mask_zero=True,)(input)
output = add_timing_signal_1d(x = embedded)
print(output)

model = tf.keras.Model(inputs = input ,outputs = output)
model.summary()
"""

