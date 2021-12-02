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

    # split data
    seed = 7
    np.random.seed(seed)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
        test_size = 0.15, random_state = seed)
    
    return x_train, y_train, x_test, y_test


# script
para = Settings()
para.number_of_states = 4
para.number_of_gaussian = 5
para.number_of_iteration = 5
para.dimension_of_vector = 39
para.frameSize = 200
para.overlapSize = 100
para.N_mel_dct = 13
para.N_mel = 26
para.training_file_directory = '..\\Audio\\train'

print("Start generating: ")
for i_word in range(10):
    x_train, y_train, x_test, y_test = load_data(i_word, para)
    np.savetxt(".\\dnn_data\\"+str(i_word)+"x_train.txt", x_train, 
        fmt='%f', delimiter=',')
    np.savetxt(".\\dnn_data\\"+str(i_word)+"y_train.txt", y_train, 
        fmt='%f', delimiter=',')
    np.savetxt(".\\dnn_data\\"+str(i_word)+"x_test.txt", x_test, 
        fmt='%f', delimiter=',')
    np.savetxt(".\\dnn_data\\"+str(i_word)+"y_test.txt", y_test, 
        fmt='%f', delimiter=',')
    print("data for " + str(i_word) + " is generated")



