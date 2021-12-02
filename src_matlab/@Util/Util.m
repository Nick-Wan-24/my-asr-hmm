classdef Util
    methods
        function obj = Util() % default constructor
        end
    end
    methods (Static)
        
        % assign T vectors to n states averagely and 
        % count MFCC number of each states 
        function [states, states_num] = distribute(states, states_num)
            T = length(states);
            nStates = size(states_num,1);
            N1 = floor(T/nStates);
            N2 = T - N1*(nStates-1);
            if (N1 < 1 || N2 < 1)
                return;
            end
            % states sequence
            for i = 1 : nStates-1
                states((i-1)*N1+1 : i*N1) = i;
                states_num(i,i) = states_num(i,i) + N1;
                states_num(i,i+1) = states_num(i,i+1) + 1;
            end
            states((nStates-1)*N1+1 : end) = nStates;
            states_num(nStates,nStates) = states_num(nStates,nStates) + N2;
            states_num(nStates,nStates+1) = states_num(nStates,nStates+1) + 1;
        end
        
        % classify the MFCC vectors to N classes using K-means
        function feats_all = kmeans(feats, N)
            ndim = size(feats, 1);
            nfeats = size(feats, 2);
            mu = zeros(ndim, N);
            feats_all = cell(1, N);
            % initialize N centers
            tmp = rand(1, N);
            for i = 1 : N
                mu(:, i) = feats(:, ceil(tmp(i)*nfeats));
            end
            mu_pre = mu;
            % iteration
            iter = 0;
            while 1
                iter = iter + 1;
                % compute distribution
                dist = feats' * mu; % dist(i,j): D^2 of mu(j) and feats(i)
                [~, feats_class] = min(dist,[],2); % class number of feats(i)
                feats_class = feats_class';
                % update mu
                for i = 1 : N
                    mu(:, i) = mean(feats(:,feats_class==i), 2);
                end
                % termination?
                if max(max(abs(mu-mu_pre))) < 1e-4 || iter > 99
                    break;
                end
                mu_pre = mu;
            end
            % write to result
            for i = 1 : N
                feats_all{i} = feats(:, feats_class==i);
            end
        end
        
        function Y = log(X)
            Y = zeros(size(X));
            Y(X ~= 0) = log(X(X ~= 0));
            Y(X == 0) = -1e20;
        end
        
        function Y = logsumexp(X)
            c = max(X);
            Y = log(sum(exp(X-c))) + c;
        end
        
        % extract features of all wav files for number 'i' in wav_dir
        function feats = feature_extract(wav_dir, para, i)
            namelist = dir([wav_dir, '\',num2str(i),'*.wav']);
            n_file = length(namelist);
            feats = cell(1, n_file);
            for f = 1 : n_file
                feats{f} = Util.mfcc([wav_dir, '\', namelist(f).name], para);
            end
        end
        
        % extract MFCC matrix of a single wav file
        function feats = mfcc(wavfile, para)
            % parameters
            frameSize = para.frameSize;
            overlapSize = para.overlapSize;
            T = 0; % for EPD; usually 1e-3;wish no EPD? set T = 0
            N_mel_dct = para.N_mel_dct;
            N_mel = para.N_mel;
            
            % load .wav file
            [x, fs] = audioread(wavfile);
            x = x(:, 1);
            % x = x / max(abs(x));
            N = length(x);
            
            % pre-process
            y = x;
            y(2 : end) = x(2 : end) - 0.97 * x(1 : end-1);
            
            % segment
            N_frame = floor((N - overlapSize) / (frameSize - overlapSize));
            N = N_frame * (frameSize - overlapSize) + overlapSize;
            y = y(1 : N);
            x = zeros(frameSize, N_frame);
            for i = 1 : N_frame
                i1 = (i-1) * (frameSize - overlapSize) + 1;
                i2 = i1 + frameSize - 1;
                x(:, i) = y(i1 : i2);
            end
            
            % EPD
            E = sum(x.^2);
            tmp = find(E > T);
            if isempty(tmp)
                error('no valid speech frame found.');
            end
            i_start = tmp(1);
            i_end = tmp(end);
            x = x(:, i_start : i_end);
            N_frame = size(x, 2);
            
            % window
            W = 0.54 - 0.46 * cos(2 * pi * (1 : frameSize) / (frameSize - 1));
            W = repmat(W', 1, N_frame);
            x = x .* W;
            
            % energy of FFT
            x = abs(fft(x)).^2 / frameSize;
            
            % Mel frequencies
            f_max = 2595 * log10(1 + fs / 2 / 700);
            f_mel = linspace(0, f_max, N_mel + 2);
            f_mel = 700 * (10 .^ (f_mel / 2595) - 1);
            
            % Mel filter
            H = zeros(frameSize, N_mel);
            f = 0 : (fs / frameSize) : (frameSize - 1) * (fs / frameSize);
            index = round(f_mel / (fs / frameSize) + 1);
            for i = 2 : (length(f_mel) - 1)
                i_now = index(i);
                i_pre = index(i - 1);
                i_next = index(i + 1);
                H(i_pre:(i_now-1), i-1) = 2*(f(i_pre:i_now-1)-f(i_pre))/...
                    (f(i_next)-f(i_pre))/(f(i_now) - f(i_pre));
                H(i_now:i_next, i-1) = 2*(f(i_next)-f(i_now:i_next))/...
                    (f(i_next)-f(i_pre))/(f(i_next)-f(i_now));
            end
            
            % Mel coefficients
            mfcc_tmp = zeros(N_mel, N_frame);
            for i = 1 : N_frame
                tmp = repmat(x(:, i), 1, N_mel);
                tmp = sum(H .* tmp);
                mfcc_tmp(:, i) = tmp';
            end
            
            % dct matrix
            dct_matrix = zeros(N_mel_dct, N_mel);
            for n = 1 : N_mel_dct
                dct_matrix(n,:) = sqrt(2/N_mel) * cos(n*pi/N_mel*((1:N_mel)-0.5));
            end
            
            % dct transformation and cepstrum
            mfcc = zeros(N_mel_dct, N_frame);
            K = 1 + (N_mel-2) / 2 * sin((1 : N_mel_dct)' * pi / (N_mel-2));
            for i = 1 : N_frame
                mfcc(:, i) = dct_matrix * log(mfcc_tmp(:, i));
                mfcc(:, i) = mfcc(:, i) .* K;
            end
            
            % derivative
            dmfcc = [mfcc(:,2) - mfcc(:,1),...
                mfcc(:,3:end) - mfcc(:,1:end-2),...
                mfcc(:,end) - mfcc(:,end-1)];
            d2mfcc = [dmfcc(:,2) - dmfcc(:,1),...
                dmfcc(:,3:end) - dmfcc(:,1:end-2),...
                dmfcc(:,end) - dmfcc(:,end-1)];
            
            % Construct
            feats = [mfcc; dmfcc; d2mfcc];
        end

        % compute Edit Distance between two vectors using DP
        function dist = distance(guess, correct)
            N1 = length(guess);
            N2 = length(correct);
            if N1 == 0 || N2 == 0
                dist = N1 + N2;
                return;
            end
            dp = zeros(N1, N2);
            dp(1,1) = (correct(1) ~= guess(1));
            tmp = 0;
            for i = 2 : N1
                if guess(i) == correct(1) && tmp == 0
                    dp(i,1) = dp(i-1,1);
                    tmp = 1;
                else
                    dp(i,1) = dp(i-1,1) + 1;
                end
            end
            tmp = 0;
            for j = 2 : N2
                if guess(1) == correct(j) && tmp == 0
                    dp(1,j) = dp(1,j-1);
                    tmp = 1;
                else
                    dp(1,j) = dp(1,j-1) + 1;
                end
            end
            for i = 2 : N1
                for j = 2 : N2
                    dp(i,j) = min([dp(i-1,j-1) + (correct(j)~=guess(i)) ...
                                   dp(i-1,j) + 1 ...
                                   dp(i,j-1) + 1]);
                end
            end
            dist = dp(end,end);
        end
    end
    
end