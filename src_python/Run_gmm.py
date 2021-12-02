# script for 0-9 digits recognize using GMM-HMM model
# use 39 length MFCC as feature
# all GMM_HMM model has the same number of states and gaussian model
# training method: viterbi + updating by EM(for multi-GMM) 
# or MLE(for single-GMM) + counting(for HMM)
# continuous pred method: optimal states computation

from PKGS.GMM_HMM import GMM_HMM
from PKGS.Settings import Settings
from PKGS.Utils import *

# loading parameters
# for GMM-HMM model
para = Settings()
para.number_of_states = 3
para.number_of_gaussian = 3
para.number_of_iteration = 2
# for MFCC computation
para.dimension_of_vector = 39
para.frameSize = 200
para.overlapSize = 100
para.N_mel_dct = 13
para.N_mel = 26
# Audio file
para.training_file_directory = '..\\Audio\\train'
para.testing_file_directory = '..\\Audio\\test'
nWords = 10


# train
print("Start training: ")
model_all = []
i_word = 0
for i_word in range(nWords):
    feats_train = feature_extract(para.training_file_directory, \
        para, i_word)
    model = GMM_HMM(para)
    while (1):
        model.initialize(feats_train)
        model.train(feats_train)
        if (model.check() == True):
            break
    model_all.append(model)
    print("model " + str(i_word) + " is trained")


# test
print("Start testing: ")
Acc = np.zeros(nWords)
for i in range(nWords):
    feats_test = feature_extract(para.testing_file_directory, \
        para, i)
    n_file = len(feats_test)
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

