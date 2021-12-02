from keras.models import load_model
import numpy as np

dnn_model = load_model(".\\dnn_model\\model_0.h5")
x_test = np.loadtxt(".\\dnn_data\\0x_test.txt", delimiter=',')

tmp = dnn_model.predict(x_test[0:10])

print(tmp)