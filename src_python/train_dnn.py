import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from PKGS.Utils import *


i = 9;

# laod data
x_train = np.loadtxt(".\\dnn_data\\"+str(i)+"x_train.txt", delimiter=',')
y_train = np.loadtxt(".\\dnn_data\\"+str(i)+"y_train.txt", delimiter=',')
x_test = np.loadtxt(".\\dnn_data\\"+str(i)+"x_test.txt", delimiter=',')
y_test = np.loadtxt(".\\dnn_data\\"+str(i)+"y_test.txt", delimiter=',')

# define model
input_dim = np.size(x_train, 1)
output_dim = np.size(y_train, 1)
model = Sequential()
model.add(Dense(39, activation = 'relu', input_dim = input_dim))
# model.add(Dense(20, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
# model.add(Dense(10, activation = 'relu'))
model.add(Dense(output_dim, activation = 'softmax')) # softmax or sigmoid
# sgd = SGD(lr = 7e-4, decay = 1e-3, momentum = 0.92, nesterov = True)
model.compile(optimizer = 'adam',
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

# train model
print("Start training DNN model " + str(i))
model.fit(x_train, 
            y_train, 
            batch_size = 300,
            epochs = 60,
            validation_data = (x_test, y_test))

# save
# model.save(".\\dnn_model\\model_" + str(i) + ".h5")
model.save("model_" + str(i) + ".h5")
# model = keras.models.load_model('my_model.h5')



