function obj = init(obj, feats)

n_file = length(feats);
nStates = obj.nStates;
nGaussian = obj.nGaussian;

% initialize pi
obj.Pi(1) = 1;
for i = 1 : nStates
    obj.pik{i} = obj.pik{i} + 1 / nGaussian;
end

% initialize states and A
states = cell(1, n_file);
if nStates == 1 % prue GMM, no need to distribute
    for f = 1 : n_file
        T = size(feats{f}, 2);
        states{f} = zeros(1, T) + 1;
    end
else
    states_num = zeros(nStates, nStates+1);
    for f = 1 : n_file
        T = size(feats{f}, 2);
        states{f} = zeros(1, T);
        [states{f}, states_num] = Util.distribute(states{f}, states_num);
    end
    for i = 1 : nStates
        obj.A(i,i) = states_num(i,i) / (states_num(i,i) + n_file);
        if i < nStates
            obj.A(i,i+1) = 1 - obj.A(i,i);
        end
    end
end

% initialize mu and sig using K-means
for i = 1 : nStates
    % extract MFCC in i state
    feats_all = [];
    for f = 1 : n_file
        feats_all = [feats_all, feats{f}(:,states{f}==i)];
    end

    % classify features in cell(1,ngaussian)
    feats_all = Util.kmeans(feats_all, nGaussian);

    % set mu and sig
    for g = 1 : nGaussian
        obj.mu{i}(:,g) = mean(feats_all{g}, 2);
        T = size(feats_all{g},2);
        obj.sig{i}(:,g) = mean((feats_all{g} - repmat(obj.mu{i}(:,g),1,T)).^2, 2);
    end
end

end