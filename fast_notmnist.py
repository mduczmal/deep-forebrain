import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, Flatten,\
MaxPooling2D, Dropout

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

#Create deep learning architecture
model = Sequential()
model.add(Convolution2D(16, 5, 5, border_mode='valid', input_shape=(28, 28, 1)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(16, 5, 5))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Activation("relu"))
model.add(Dense(10))
model.add(Activation("softmax"))

#Train the network and evaluate on the test set
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(train_dataset, train_labels, nb_epoch=5, batch_size=batch_size)
loss_and_metrics = model.evaluate(test_dataset, test_labels,
                                  batch_size=batch_size)

#Show outcomes
print("Test_score: {}".format(loss_and_metrics[0]))
print("Test_accuracy: {}%".format(loss_and_metrics[1]*100))
# 0.179573610249, 94.9%
