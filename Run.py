# script for 0-9 digits recognize using GMM-HMM model
# use 39 length MFCC as feature
# all GMM_HMM model has the same number of states and gaussian model
# training method: viterbi + updating by EM(for multi-GMM) 
# or MLE(for single-GMM) + counting(for HMM)
# continuous pred method: optimal states computation

from GMM_HMM import GMM_HMM
from Settings import Settings
from Utils import *

# loading parameters
para = Settings()
nWords = 10

# train
model_all = []
i_word = 0
while (i_word < nWords):
    feats_train = feature_extract(para.training_file_directory, \
        para, i_word)
    model = GMM_HMM(para)
    model.initialize(feats_train)
    model.train(feats_train)
    if (model.check() == False):
        continue
    model_all.append(model)
    i_word += 1

# test
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
    print('Accuracy of '+str(i)+' is '+str(Acc[i])+'%')










