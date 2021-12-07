% script for 0-9 digits recognize using GMM-HMM model
% use 39 length MFCC as feature
% all GMM_HMM model has the same number of states and gaussian model
% training method: viterbi + updating by EM(for multi-GMM) 
% or MLE(for single-GMM) + counting(for HMM)
% continuous pred method: optimal states computation

% loading parameters
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
training_file_directory = '..\Audio\train';
testing_file_directory = '..\Audio\test';
if if_continuous == 1
    testing_file_directory = [testing_file_directory '_continuous'];
end
nWords = 10;


%% train
model_all = cell(1, nWords);
i_word = 0;
for i_word = 0 : nWords-1
    feats_train = Util.feature_extract(training_file_directory, para, i_word);
    model = GMM_HMM(para);
    while 1
        model = model.init(feats_train);
        model = model.train(feats_train);
        if (model.check()) % training error: NaN parameter
            break; % or, repeat training
        end
    end
    model_all{i_word+1} = model;
end
if if_continuous == 1
    model = GMM_HMM.joint(model_all);
end


%% test
if if_continuous == 0 % for discretization digits
    Acc = zeros(1, nWords);
    for i = 0 : nWords-1
        feats_test = Util.feature_extract(testing_file_directory, para, i);
        n_file = length(feats_test);
        ll = zeros(nWords, n_file);
        for j = 0 : nWords-1
            ll(j+1,:) = model_all{j+1}.compute_ll(feats_test);
        end
        [~,result] = max(ll);
        correct = 0;
        for j = 1 : n_file
            if result(j)-1 == i
                correct = correct + 1;
            end
        end
        Acc(i+1) = correct / n_file * 100;
        disp(['Accuracy of ' num2str(i) ' is ' num2str(Acc(i+1)) '%'])
    end
else % for continuous digits
    feats_test = Util.feature_extract(testing_file_directory, para, '');
    states = model.viterbi(feats_test);
    load([testing_file_directory '\number_continuous.mat']); % load correct (11xN)
    result = cell(1, length(feats_test));
    Acc = zeros(1, length(feats_test));
    for i = 1 : length(feats_test)
        result{i} = GMM_HMM.decode(states{i}, model.nStates, nWords);
        Acc(i) = Util.distance(result{i}, correct(:,i));
    end
    disp(['Average edit distance of continuous audio is ' num2str(mean(Acc))])
end

