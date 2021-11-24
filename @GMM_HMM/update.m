function obj = update(obj, feats, states)

nStates = obj.nStates;
n_file = length(states);
nGaussian = obj.nGaussian;

% update A through counting
states_num = zeros(nStates, nStates+1);
for f = 1 : n_file
    for i = 1 : nStates
        states_num(i,i) = states_num(i,i) + length(find(states{f} == i));
        states_num(i,i+1) = states_num(i,i+1) + 1;
    end
end
for i = 1 : nStates
    obj.A(i,:) = states_num(i,:) / sum(states_num(i,:));
end

% update mu and sig
for i = 1 : nStates
    feats_all = [];
    for f = 1 : n_file
        feats_all = [feats_all, feats{f}(:,states{f}==i)];
    end
    
    if nGaussian > 1 % for muti-GMM, EM algorithm
        gamma = obj.e_step(feats_all, i); % e_step
        obj = obj.m_step(feats_all, gamma, i); % m_step
    else % for single gaussian, MLE
        model.mu{i} = mean(feats_all, 2);
        model.sig{i} = mean((feats_all - repmat(model.mu{i},1,size(feats_all,2))).^2, 2);
    end
end

end