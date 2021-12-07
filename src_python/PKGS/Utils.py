import numpy as np
import random as rd
import wave
import os


# assign T vectors to n states averagely and 
# count MFCC number of each states 
def distribute(states, states_num):
    T = np.size(states)
    nStates = np.size(states_num, 0)
    N1 = int(np.floor(T / nStates))
    N2 = int(T - N1 * (nStates-1))
    if (N1 < 1 or N2 < 1):
        return states, states_num
    for i in range(nStates-1):
        states[i*N1 : (i+1)*N1] = i + 1
        states_num[i][i] += N1
        states_num[i][i+1] += 1
    states[(nStates-1)*N1 : ] = nStates
    states_num[-1][-2] += N2
    states_num[-1][-1] += 1
    return states-1, states_num


# classify the MFCC vectors to N classes using K-means
def kmeans(feats, N):
    nDim = np.size(feats, 1)
    nfeats = np.size(feats, 0)
    mu = np.zeros((N,nDim))
    
    # initialize N centers
    for i in range(N):
        index = rd.randint(0, nfeats-1)
        mu[i] = feats[index]
    
    # iteration
    mu_pre = mu
    iter = 0
    while (1):
        iter += 1
        # compute distribution
        dist = np.matmul(feats, mu.T) # dist[i][j]: D^2 of mu[j] and feats[i]
        feats_class = np.argmin(dist, axis=1) # f[i]: class number of feats[i]
        # update mu
        for i in range(N):
            mu[i] = np.mean(feats[feats_class==i][:], axis=0)
        # termination?
        if (np.max(np.abs(mu - mu_pre)) < 1e-4 or iter > 99):
            break
        mu_pre = mu
    # write to result
    feats_all = []
    for i in range(N):
        feats_all.append(feats[feats_class==i][:])
    
    return feats_all


def elog(X):
    Y = np.zeros_like(X)
    Y[X != 0] = np.log(X[X != 0])
    Y[X == 0] = -1e20
    return Y

def logsumexp(X):
    c = np.max(X)
    Y = np.log(np.sum(np.exp(X-c))) + c
    return Y


def feature_extract(wavDir, para, i):
    file_all = os.listdir(wavDir)
    feats = []
    for file in file_all:
        if (file.startswith(str(i))):
            feats.append(mfcc(wavDir+'\\'+file, para))
    return feats


def mfcc(wavfile, para):
    frameSize = para.frameSize
    overlapSize = para.overlapSize
    N_mel_dct = para.N_mel_dct
    N_mel = para.N_mel

    # read data to x and fs (Hz)
    f = wave.open(wavfile, "rb")
    params = f.getparams()
    fs, N = params[2:4]
    tmp = f.readframes(N)
    f.close()
    x = np.frombuffer(tmp, dtype = np.int16)
    x = x.astype(np.float64) / 32768 # range: -1 to 1

    # pre-process
    y = x
    y[1:] = x[1:] - 0.97 * x[0:-1]

    # segment
    N_frame = int(np.floor((N-overlapSize) / (frameSize-overlapSize)))
    N = N_frame * (frameSize-overlapSize) + overlapSize
    y = y[:N]
    x = np.zeros((N_frame, frameSize), dtype = np.float64)
    for i in range(N_frame):
        i1 = i * (frameSize - overlapSize)
        x[i] = y[i1 : (i1+frameSize)]

    # window
    tmp = np.arange(1, frameSize+1)
    W = 0.54 - 0.46*np.cos(2*np.pi*tmp / (frameSize-1))
    x = np.multiply(x, W)

    # energy of FFT
    x = np.fft.fft(x, axis = 1)
    x = np.abs(np.multiply(x,x)) / frameSize

    # mel frequencies
    f_max = 2595 * np.log10(1 + fs/2/700)
    f_mel = np.linspace(0, f_max, N_mel+2)
    f_mel = 700 * (np.power(10,f_mel/2595) - 1)

    # mel filter
    H = np.zeros((N_mel, frameSize), dtype = np.float64)
    f = np.linspace(0, (frameSize-1)*(fs/frameSize), frameSize)
    index = np.arange(N_mel+2, dtype = np.int16)
    for i in range(0, N_mel+2):
        index[i] = round(f_mel[i] / (fs/frameSize) + 1) - 1
    for i in range(1, N_mel+1):
        i_now = index[i]
        i_pre = index[i-1]
        i_next = index[i+1]
        H[i-1][i_pre:i_now] = 2*(f[i_pre:i_now]-f[i_pre]) / (f[i_next]-f[i_pre]) \
            / (f[i_now]-f[i_pre])
        H[i-1][i_now:i_next+1] = 2*(f[i_next]-f[i_now:i_next+1]) \
            / (f[i_next]-f[i_pre]) / (f[i_next]-f[i_now])

    # mel coefficients
    mfcc_tmp = np.zeros((N_frame, N_mel), dtype = np.float64)
    for i in range(0, N_frame):
        for j in range(0, N_mel):
            mfcc_tmp[i][j] = np.dot(x[i], H[j])

    # dct matrix
    dct_matrix = np.zeros((N_mel_dct, N_mel), dtype = np.float64)
    tmp = np.pi / N_mel * (np.arange(1,N_mel+1) - 0.5)
    for i in range(0, N_mel_dct):
        dct_matrix[i] = np.sqrt(2/N_mel) * np.cos((i+1) * tmp)

    # dct transformation and cepstrum
    mfcc = np.zeros((N_frame, N_mel_dct), dtype = np.float64)
    tmp = np.arange(1, N_mel_dct+1) * np.pi / (N_mel-2)
    K = 1 + (N_mel-2) / 2 * np.sin(tmp)
    for i in range(0, N_frame):
        log_mfcc_tmp = np.log(mfcc_tmp[i])
        for j in range(0, N_mel_dct):
            mfcc[i][j] = np.dot(dct_matrix[j], log_mfcc_tmp)
        mfcc[i] = np.multiply(mfcc[i], K)

    # derivative
    dmfcc = np.zeros_like(mfcc)
    dmfcc[1:-1][:] = mfcc[2:][:] - mfcc[0:-2][:]
    dmfcc[0][:] = mfcc[1][:] - mfcc[0][:]
    dmfcc[-1][:] = mfcc[-1][:] - mfcc[-2][:]
    d2mfcc = np.zeros_like(dmfcc)
    d2mfcc[1:-1][:] = dmfcc[2:][:] - dmfcc[0:-2][:]
    d2mfcc[0][:] = dmfcc[1][:] - dmfcc[0][:]
    d2mfcc[-1][:] = dmfcc[-1][:] - dmfcc[-2][:]

    # construct
    feats = np.concatenate((mfcc,dmfcc), axis = 1)
    feats = np.concatenate((feats,d2mfcc), axis = 1)

    return feats
    

def feature_extract_dnn(wavDir, para, i):
    file_all = os.listdir(wavDir)
    feats = []
    for file in file_all:
        if (file.startswith(str(i))):
            feats_tmp = fbank(wavDir+'\\'+file, para)
            feats.append(feats_tmp)
    return feats


def fbank(wavfile, para):
    frameSize = para.frameSize
    overlapSize = para.overlapSize
    N_mel = para.N_mel

    # read data to x and fs (Hz)
    f = wave.open(wavfile, "rb")
    params = f.getparams()
    fs, N = params[2:4]
    tmp = f.readframes(N)
    f.close()
    x = np.frombuffer(tmp, dtype = np.int16)
    x = x.astype(np.float64) / 32768 # range: -1 to 1

    # pre-process
    y = x
    y[1:] = x[1:] - 0.97 * x[0:-1]

    # segment
    N_frame = int(np.floor((N-overlapSize) / (frameSize-overlapSize)))
    N = N_frame * (frameSize-overlapSize) + overlapSize
    y = y[:N]
    x = np.zeros((N_frame, frameSize), dtype = np.float64)
    for i in range(N_frame):
        i1 = i * (frameSize - overlapSize)
        x[i] = y[i1 : (i1+frameSize)]

    # window
    tmp = np.arange(1, frameSize+1)
    W = 0.54 - 0.46*np.cos(2*np.pi*tmp / (frameSize-1))
    x = np.multiply(x, W)

    # energy of FFT
    x = np.fft.fft(x, axis = 1)
    x = np.abs(np.multiply(x,x)) / frameSize

    # mel frequencies
    f_max = 2595 * np.log10(1 + fs/2/700)
    f_mel = np.linspace(0, f_max, N_mel+2)
    f_mel = 700 * (np.power(10,f_mel/2595) - 1)

    # mel filter
    H = np.zeros((N_mel, frameSize), dtype = np.float64)
    f = np.linspace(0, (frameSize-1)*(fs/frameSize), frameSize)
    index = np.arange(N_mel+2, dtype = np.int16)
    for i in range(0, N_mel+2):
        index[i] = round(f_mel[i] / (fs/frameSize) + 1) - 1
    for i in range(1, N_mel+1):
        i_now = index[i]
        i_pre = index[i-1]
        i_next = index[i+1]
        H[i-1][i_pre:i_now] = 2*(f[i_pre:i_now]-f[i_pre]) / (f[i_next]-f[i_pre]) \
            / (f[i_now]-f[i_pre])
        H[i-1][i_now:i_next+1] = 2*(f[i_next]-f[i_now:i_next+1]) \
            / (f[i_next]-f[i_pre]) / (f[i_next]-f[i_now])

    # mel coefficients
    mfcc = np.zeros((N_frame, N_mel), dtype = np.float64)
    for i in range(0, N_frame):
        for j in range(0, N_mel):
            mfcc[i][j] = np.log(np.dot(x[i], H[j]))

    # derivative
    dmfcc = np.zeros_like(mfcc)
    dmfcc[1:-1][:] = mfcc[2:][:] - mfcc[0:-2][:]
    dmfcc[0][:] = mfcc[1][:] - mfcc[0][:]
    dmfcc[-1][:] = mfcc[-1][:] - mfcc[-2][:]
    d2mfcc = np.zeros_like(dmfcc)
    d2mfcc[1:-1][:] = dmfcc[2:][:] - dmfcc[0:-2][:]
    d2mfcc[0][:] = dmfcc[1][:] - dmfcc[0][:]
    d2mfcc[-1][:] = dmfcc[-1][:] - dmfcc[-2][:]

    # construct
    feats = np.concatenate((mfcc,dmfcc), axis = 1)
    feats = np.concatenate((feats,d2mfcc), axis = 1)
    
    return feats

def normalize(feats, mu, sig):
    feats = (feats - mu) / np.sqrt(sig+1e-6)
    return feats