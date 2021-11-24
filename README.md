# my-asr-hmm

### 0-9 digits recognition based on GMM-HMM model and MFCC

#### **Audio file source:**

[GitHub - Jakobovski/free-spoken-digit-dataset: A free audio dataset of spoken digits. Think MNIST for audio.](https://github.com/Jakobovski/free-spoken-digit-dataset)

#### **Training method:**

[GMM-HMM]

- We initialize the model through averagely distribute MFCCs to states and use K-means to distribute MFCCs in one state again to gaussian models.

- Viterbi algorithm is used in E-step to get optimal states sequence and update A matrix in HMM by directly counting
- EM algorithm is used for GMM updating

[GMM]

- Initialize model by all .wav files and K-means
- Update GMM by traditional EM algorithm

[HMM]

- Initialize model only by distribution and counting
- Viterbi algorithm is used in E-step to get optimal states sequence and update A matrix in HMM by directly counting
- MLE is used for gaussian model updating

