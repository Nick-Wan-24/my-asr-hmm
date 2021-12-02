import numpy as np
from PKGS.GMM_HMM import GMM_HMM
from PKGS.Utils import *
from keras.models import load_model

class DNN_HMM:

    def __init__(self, para, i):
        feats_train = feature_extract(para.training_file_directory, \
            para, i)
        self.gmm_model = GMM_HMM(para)
        while (1):
            self.gmm_model.initialize(feats_train)
            self.gmm_model.train(feats_train)
            if (self.gmm_model.check() == True):
                break
        tmp = np.loadtxt(".\\dnn_data\\"+str(i)+"y_train.txt",delimiter=',')
        self.Ps = np.sum(tmp, axis=0) / np.size(tmp, axis=0)
        self.dnn_model = load_model('.\\dnn_model\\model_'+str(i)+'.h5')


    # functions:
    # logb = compute_logb(self, feats)
    # ll = compute_ll(self, feats)


    def compute_logb(self, feats):
        nStates = self.gmm_model.nStates
        T = np.size(feats, 0)

        # matrix allocation
        logPos = np.zeros((T, nStates))

        # DNN prediction
        Pso = self.dnn_model.predict(feats)

        # log
        logPso = elog(Pso)
        logPs = elog(self.Ps)
        
        # compute b
        for t in range(T):
            logPos[t] = logPso[t] - logPs
        logb = logPos.T

        return logb
    

    def compute_ll(self, feats):
        nStates = self.gmm_model.nStates
        n_file = len(feats)
        
        ll = np.zeros(n_file)
        for f in range(n_file):
            # compute logb matrix
            logb = self.compute_logb(feats[f])

            # compute likelihood, no need to forward
            if (nStates < 2):
                ll[f] = np.sum(logb)
                continue
                
            # set A and pi also to log value
            logA = elog(self.gmm_model.A)
            logPi = elog(self.gmm_model.Pi)

            # compute logdelta and phi
            T = np.size(feats[f], 0)
            logalpha = np.zeros((nStates, T))
            for i in range(nStates):
                logalpha[i][0] = logPi[i] + logb[i][0]
            alpha_tmp = np.zeros(nStates)
            A_tmp = np.zeros(nStates)
            for t in range(1,T):
                for i in range(nStates):
                    alpha_tmp[i] = logalpha[i][t-1]
                for i in range(nStates):
                    for j in range(nStates):
                        A_tmp[j] = logA[j][i]
                    tmp = A_tmp + alpha_tmp
                    logalpha[i][t] = logb[i][t] + logsumexp(tmp)

            # compute likelihood
            alpha_tmp = np.zeros(nStates)
            for i in range(nStates):
                alpha_tmp[i] = logalpha[i][-1]
            ll[f] = logsumexp(alpha_tmp)
        
        return ll
        





