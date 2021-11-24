% for GMM-HMM model
para.number_of_states = 4; % for prue GMM, set to 1
para.number_of_gaussian = 5; % for HMM with single GMM, set to 1
para.number_of_iteration = 5; % 5 to 10 is recommended

% for MFCC computation
para.dimension_of_vector = 39;
para.frameSize = 200;
para.overlapSize = 100;
para.N_mel_dct = 13;
para.N_mel = 26;

% 0 for discretization, 1 for continuous
if_continuous = 1;

% Audio file
training_file_directory = '.\Audio\train';
testing_file_directory = '.\Audio\test';
if if_continuous == 1
    testing_file_directory = [testing_file_directory '_continuous'];
end