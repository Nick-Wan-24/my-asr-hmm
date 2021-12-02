function obj = train(obj, feats)
% feats is a Nx1 cell, each elements is a 39xT MFCC matrix in a file

if obj.nStates > 1 % viterbi (for HMM)
    for i = 1 : obj.nIter
        states = obj.viterbi(feats);
        obj = obj.update(feats, states);
    end
else % EM algorithm (for GMM)
    feats_all = [];
    for f = 1 : length(feats)
        feats_all = [feats_all, feats{f}];
    end
    for i = 1 : obj.nIter
        gamma = obj.e_step(feats_all, 1);
        obj = obj.m_step(feats_all, gamma, 1);
    end
end

end