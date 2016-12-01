import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten,\
MaxPooling2D, AveragePooling2D, Dropout, Merge

image_size = 28
num_labels = 10
num_channels = 1 # grayscale
batch_size = 16
pickle_file = '/home/guzik/programms/DeepDream/notMNIST.pickle'

#Load notMNIST dataset
with open(pickle_file, 'rb') as f:
  save = pickle.load(f);
  pre_train_dataset = save['train_dataset']
  pre_train_labels = save['train_labels']
  pre_valid_dataset = save['valid_dataset']
  pre_valid_labels = save['valid_labels']
  pre_test_dataset = save['test_dataset']
  pre_test_labels = save['test_labels']
  del save;  # hint to help gc free up memory
  print('Training set', pre_train_dataset.shape, pre_train_labels.shape)
  print('Validation set', pre_valid_dataset.shape, pre_valid_labels.shape)
  print('Test set', pre_test_dataset.shape, pre_test_labels.shape)

#Put data in tensorflow-friendly NHWC format.
def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(pre_train_dataset, pre_train_labels)
valid_dataset, valid_labels = reformat(pre_valid_dataset, pre_valid_labels)
test_dataset, test_labels = reformat(pre_test_dataset, pre_test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

#Create quasi inception module based on GoogLeNet
#First inception module
branch_11 = Sequential()
branch_11.add(Convolution2D(4, 1, 1, border_mode='same', input_shape=(28, 28, 1)))
branch_11.add(Activation("relu"))
branch_13 = Sequential()
branch_13.add(Convolution2D(8, 3, 3, border_mode='same', input_shape=(28, 28, 1)))
branch_13.add(Activation("relu"))
branch_15 = Sequential()
branch_15.add(Convolution2D(16, 5, 5, border_mode='same', input_shape=(28, 28, 1)))
branch_15.add(Activation("relu"))
merged1 = Merge([branch_11, branch_13, branch_15], mode='concat')

pool1 = Sequential()
pool1.add(merged1)
pool1.add(MaxPooling2D(pool_size=(2, 2), border_mode='valid'))

#Second inception module
branch_21 = Sequential()
branch_21.add(pool1)
branch_21.add(Convolution2D(32, 1, 1, border_mode='same'))
branch_21.add(Activation("relu"))
branch_23 = Sequential()
branch_23.add(pool1)
branch_23.add(Convolution2D(48, 3, 3, border_mode='same'))
branch_23.add(Activation("relu"))
branch_25 = Sequential()
branch_25.add(pool1)
branch_25.add(Convolution2D(64, 5, 5, border_mode='same'))
branch_25.add(Activation("relu"))
merged2 = Merge([branch_21, branch_23, branch_25], mode='concat')

#Last processing stage
model = Sequential()
model.add(merged2)
model.add(Flatten())
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))

#Train the network and evaluate on the test set
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit([train_dataset, train_dataset, train_dataset], train_labels, nb_epoch=5, batch_size=batch_size)
loss_and_metrics = model.evaluate([test_dataset, test_dataset, test_dataset], test_labels,
                                  batch_size=batch_size)

#Show outcomes
print("Test_score: {}".format(loss_and_metrics[0]))
print("Test_accuracy: {}%".format(loss_and_metrics[1]*100))
#0.130697919896, 96.24%
