# script for 0-9 digits recognize using DNN-HMM model
# use 26 length FBank as feature
# DNNs are trained by viterbi states from trained GMM_HMMs
# P(ot|si) is computed based on DNN output: = P(si|ot) / P(si)
# P(si) is computed using directly counting using training data

from PKGS.DNN_HMM import DNN_HMM
from PKGS.Settings import Settings
from PKGS.Utils import *


# loading parameters
# for GMM-HMM model
para = Settings()
para.number_of_states = 3
para.number_of_gaussian = 4
para.number_of_iteration = 10
# for feature computation
para.dimension_of_vector = 39
para.frameSize = 200
para.overlapSize = 100
para.N_mel_dct = 13
para.N_mel = 26
# Audio file
para.training_file_directory = '..\\Audio\\train'
para.testing_file_directory = '..\\Audio\\train'
nWords = 10


# initialize
print("Start initializing: ")
model_all = []
i_word = 0
for i_word in range(nWords):
    model = DNN_HMM(para, i_word)
    model_all.append(model)
    print("model " + str(i_word) + " is initialized")


# test
print("Start testing: ")
Acc = np.zeros(nWords)
for i in range(nWords):
    feats_test = feature_extract_dnn(para.testing_file_directory, \
        para, i)
    n_file = len(feats_test)
    mu = np.loadtxt(".\\dnn_data\\"+str(i)+"mu.txt", delimiter=',')
    sig = np.loadtxt(".\\dnn_data\\"+str(i)+"sig.txt", delimiter=',')
    for f in range(n_file):
        feats_test[f] = normalize(feats_test[f], mu, sig)
    ll = np.zeros((nWords, n_file))
    for j in range(nWords):
        ll[j] = model_all[j].compute_ll(feats_test)
    result = np.argmax(ll, axis=0)
    correct = 0.
    for f in range(n_file):
        if (result[f] == i):
            correct += 1
    Acc[i] = correct / n_file * 100
    print("Accuracy of " + str(i) + " is " + str(Acc[i]) + "%")


