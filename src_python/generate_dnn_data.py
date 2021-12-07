import numpy as np
from sklearn.model_selection import train_test_split
from PKGS.GMM_HMM import GMM_HMM
from PKGS.Settings import Settings
from PKGS.Utils import *


# load data steps (for number x):
#   1 define parameters
#   2 give x_train
#   3 train a GMM_HMM model
#   4 give y_train
#   5 split train_data and test_data
def load_data(i_word, para):

    # x_train
    feats = feature_extract_dnn(para.training_file_directory, para, i_word)
    x_train = feats[0]
    for i in range(1,len(feats)):
        x_train = np.concatenate((x_train, feats[i]), axis = 0)

    # train GMM_HMM and states
    model = GMM_HMM(para)
    feats = feature_extract(para.training_file_directory, para, i_word)
    while (1):
        model.initialize(feats)
        model.train(feats)
        if (model.check() == True):
            break
    states = model.viterbi(feats)

    # y_train
    y_train = np.zeros((1, model.nStates))
    for i in range(len(states)):
        for j in range(np.size(states[i],0)):
            tmp = np.zeros((1, model.nStates))
            tmp[0][states[i][j]] = 1
            y_train = np.concatenate((y_train, tmp), axis = 0)
    y_train = y_train[1:]

    # shuffle and split test data
    training_data = np.concatenate((x_train, y_train), axis=1)
    np.random.shuffle(training_data)
    dim_x = np.size(x_train, axis=1)
    n_feats = np.size(x_train, axis=0)
    for i in range(n_feats):
        x_train[i] = training_data[i][0:dim_x]
        y_train[i] = training_data[i][dim_x:]
    n_train = int(n_feats * 0.9)
    x_test = x_train[n_train:]
    y_test = y_train[n_train:]
    x_train = x_train[0:n_train]
    y_train = y_train[0:n_train]

    return x_train, y_train, x_test, y_test


# script
para = Settings()
para.number_of_states = 3
para.number_of_gaussian = 4
para.number_of_iteration = 10
para.dimension_of_vector = 39
para.frameSize = 200
para.overlapSize = 100
para.N_mel_dct = 13
para.N_mel = 26
para.training_file_directory = '..\\Audio\\train'

print("Start generating: ")
for i_word in range(1,10):
    x_train, y_train, x_test, y_test = load_data(i_word, para)
    mu = np.mean(x_train, axis=0)
    sig = np.mean(np.multiply(x_train-mu,x_train-mu), axis=0)
    x_train = normalize(x_train, mu, sig)
    x_test = normalize(x_test, mu, sig)
    np.savetxt(".\\dnn_data\\"+str(i_word)+"x_train.txt", x_train, 
        fmt='%f', delimiter=',')
    np.savetxt(".\\dnn_data\\"+str(i_word)+"y_train.txt", y_train, 
        fmt='%f', delimiter=',')
    np.savetxt(".\\dnn_data\\"+str(i_word)+"x_test.txt", x_test, 
        fmt='%f', delimiter=',')
    np.savetxt(".\\dnn_data\\"+str(i_word)+"y_test.txt", y_test, 
        fmt='%f', delimiter=',')
    np.savetxt(".\\dnn_data\\"+str(i_word)+"mu.txt", mu, 
        fmt='%f', delimiter=',')
    np.savetxt(".\\dnn_data\\"+str(i_word)+"sig.txt", sig, 
        fmt='%f', delimiter=',')
    print("data for " + str(i_word) + " is generated")



