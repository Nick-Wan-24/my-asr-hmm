import numpy as np
from PKGS.Utils import *

class GMM_HMM:

    def __init__(self, para):
        self.nStates = para.number_of_states
        self.nGaussian = para.number_of_gaussian
        self.nIter = para.number_of_iteration
        self.nDim = para.dimension_of_vector
        self.Pi = np.zeros(self.nStates, dtype = np.float64)
        self.A = np.zeros((self.nStates,self.nStates+1), dtype = np.float64)
        self.mu = np.zeros((self.nStates,self.nGaussian,self.nDim), \
            dtype = np.float64)
        self.sig = np.zeros_like(self.mu)
        self.pik = np.zeros((self.nStates,self.nGaussian), dtype = np.float64)


    # functions:
    # initialize(self, feats)
    # train(self, feats)
    # states = viterbi(self, feats)
    # logb = compute_logb(self, feats)
    # update(self, feats, states)
    # gamma = e_step(self, feats_all, i)
    # m_step(self, feats_all, gamma, i)
    # normal = check(self)
    # ll = compute_ll(self, feats)


    def initialize(self, feats) -> None:
        n_file = len(feats)
        nStates = self.nStates
        nGaussian = self.nGaussian
        nDim = self.nDim

        # initialize pi
        self.Pi[0] = 1
        self.pik[:][:] = 1 / nGaussian

        # initialize states and A
        states = [] # create an empty list 
        if nStates == 1:
            for f in range(n_file):
                T = np.size(feats[f], 0)
                states.append(np.zeros(T, dtype=np.int16) + 1)
        else:
            states_num = np.zeros((nStates,nStates+1), dtype=np.int16)
            for f in range(n_file):
                T = np.size(feats[f], 0)
                states.append(np.zeros(T, dtype=np.int16))
                states[f], states_num = distribute(states[f], states_num)
            for i in range(nStates):
                self.A[i][i] = states_num[i][i] / (states_num[i][i]+n_file)
                self.A[i][i+1] = 1 - self.A[i][i]
        
        # initialize mu and sig using K-means
        for i in range(nStates):
            # extract MFCC in i states
            feats_all = np.zeros((1,nDim)) # first line is for append
            for f in range(n_file):
                # states_tmp = np.tile(states[f], (nDim, 1))
                # states_tmp = states_tmp.T
                feats_all = np.concatenate((feats_all, feats[f][states[f]==i]), axis = 0)
            feats_all = feats_all[1:]

            # classify features in cell(1,ngaussian)
            feats_all = kmeans(feats_all, nGaussian) # [g]: feats in g Gaussian model

            # set mu and sig
            for g in range(nGaussian):
                self.mu[i][g] = np.mean(feats_all[g], axis=0)
                tmp = feats_all[g] - self.mu[i][g]
                self.sig[i][g] = np.mean(np.multiply(tmp,tmp), axis=0)


    def train(self, feats) -> None:
        if (self.nStates > 1): # viterbi (for HMM)
            for i in range(self.nIter):
                states = self.viterbi(feats)
                self.update(feats, states)
        else: # EM (for GMM), no need it
            feats_all = np.zeros((1,self.nDim)) # first line is for append
            for f in range(len(feats)):
                feats_all = np.concatenate((feats_all, feats[f]), axis=0)
            feats_all = feats_all[1:][:]
            for i in range(self.nIter):
                gamma = self.e_step(feats_all, 0)
                self.m_step(feats_all, gamma, 0)


    def viterbi(self, feats):
        n_file = len(feats)
        nStates = self.nStates
        
        states = []
        for f in range(n_file):
            # compute logb matrix
            logb = self.compute_logb(feats[f])

            # set A and pi also to log value
            logA = elog(self.A)
            logPi = elog(self.Pi)

            # compute logdelta and phi
            T = np.size(feats[f], 0)
            logdelta = np.zeros((nStates, T))
            phi = np.zeros_like(logdelta)
            for i in range(nStates):
                logdelta[i][0] = logPi[i] + logb[i][0]
            delta_tmp = np.zeros(nStates)
            A_tmp = np.zeros(nStates)
            for t in range(1,T):
                for i in range(nStates):
                    delta_tmp[i] = logdelta[i][t-1]
                for i in range(nStates):
                    for j in range(nStates):
                        A_tmp[j] = logA[j][i]
                    logdelta[i][t] = np.max(delta_tmp + A_tmp)
                    phi[i][t] = np.argmax(delta_tmp + A_tmp)

            # compute optimal states
            tmp = np.zeros(T, dtype = int)
            tmp[-1] = np.argmax(logdelta[-1])
            for t in range(T-1): # from T-2 to 0
                t_now = T - 2 - t
                tmp[t_now] = phi[tmp[t_now+1]][t_now+1]
            states.append(tmp)

        return states


    def compute_logb(self, feats):
        nStates = self.nStates
        T = np.size(feats, 0)
        nGaussian = self.nGaussian
        nDim = self.nDim

        # matrix allocation
        logb = np.zeros((nStates, T))
        logbk = np.zeros((nStates, T, nGaussian))
        logpik = elog(self.pik)
        
        # compute b
        tmp1 = -0.5 * nDim * np.log(2*np.pi)
        for i in range(nStates):
            if (nGaussian > 1):
                for g in range(nGaussian):
                    mu = self.mu[i][g] + 1e-6
                    sig = self.sig[i][g] + 1e-6
                    tmp2 = 0.5 * np.sum(elog(sig))
                    for t in range(T):
                        ot = feats[t]
                        tmp = np.divide(np.multiply(ot-mu,ot-mu), sig)
                        logbk[i][t][g] = tmp1 - tmp2 - 0.5*np.sum(tmp)
                for t in range(T):
                    logb[i][t] = logsumexp(logpik[i] + logbk[i][t])
            else:
                mu = self.mu[i][0] + 1e-6
                sig = self.sig[i][0] + 1e-6
                tmp2 = 0.5 * np.sum(elog(sig))
                for t in range(T):
                    ot = feats[t]
                    tmp = np.divide(np.multiply(ot-mu,ot-mu), sig)
                    logb[i][t] = tmp1 - tmp2 - 0.5*np.sum(tmp)

        return logb
    

    def update(self, feats, states) -> None:
        nStates = self.nStates
        n_file = len(states)
        nGaussian = self.nGaussian
        nDim = self.nDim

        # update A through counting
        states_num = np.zeros((nStates, nStates+1), dtype = int)
        for f in range(n_file):
            for i in range(nStates):
                states_num[i][i] += np.sum(states[f] == i)
                states_num[i][i+1] += 1
        for i in range(nStates):
            self.A[i] = states_num[i] / np.sum(states_num[i])
        
        # update mu and sig
        for i in range(nStates):
            feats_all = np.zeros((1, nDim))
            for f in range(n_file):
                feats_all = np.concatenate((feats_all, feats[f][states[f]==i]), axis=0)
            feats_all = feats_all[1:][:]
            if (nGaussian > 1): # for muti-GMM, EM algorithm
                gamma = self.e_step(feats_all, i)
                self.m_step(feats_all, gamma, i)
            else: # for single gaussian, MLE
                self.mu[i][0] = np.mean(feats_all, axis=0)
                tmp = feats_all - self.mu[i][0]
                self.sig[i][0] = np.mean(np.multiply(tmp,tmp), axis=0)
    

    def e_step(self, feats, i):
        nGaussian = self.nGaussian
        T = np.size(feats, 0)
        
        # compute pi_k * logN(t|mu_k, sig_k)
        logN = np.zeros((nGaussian, T))
        tmp1 = -0.5 * self.nDim * np.log(2*np.pi)
        for g in range(nGaussian):
            mu = self.mu[i][g] + 1e-6
            sig = self.sig[i][g] + 1e-6
            tmp2 = 0.5 * np.sum(elog(sig))
            for t in range(T):
                ot = feats[t]
                tmp = np.divide(np.multiply(ot-mu,ot-mu), sig)
                logN[g][t] = tmp1 - tmp2 - 0.5*np.sum(tmp)
        N = np.exp(logN)
        N = N.T * self.pik[i]
        N = N.T

        # compute gamma (t,g|ot,self)
        gamma = N / np.max(N,0)
        return gamma


    def m_step(self, feats, gamma, i) -> None:
        nGaussian = self.nGaussian
        nDim = self.nDim
        T = np.size(feats, 0)
        
        # updating
        for g in range(nGaussian):
            mu = np.zeros(nDim)
            sum_G = np.sum(gamma[g])
            for t in range(T):
                mu += (gamma[g][t] * feats[t])
            self.mu[i][g] = mu / (sum_G+1e-6)
            sig = np.zeros(nDim)
            for t in range(T):
                tmp = feats[t] - self.mu[i][g]
                sig += (gamma[g][t] * np.multiply(tmp,tmp))
            self.sig[i][g] = sig / (sum_G+1e-6)
            self.pik[i][g] = sum_G / T

    
    def check(self) -> bool:
        if (np.sum(np.isnan(self.mu + self.sig)) > 0):
            return False
        return True
    

    def compute_ll(self, feats):
        nStates = self.nStates
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
            logA = elog(self.A)
            logPi = elog(self.Pi)

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
        





