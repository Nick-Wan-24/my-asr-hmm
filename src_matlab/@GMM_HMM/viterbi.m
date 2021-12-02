function states = viterbi(obj, feats)

nStates = obj.nStates;
n_file = length(feats);

states = cell(1, n_file);
for f = 1 : n_file
    % compute logb matrix
    logb = obj.compute_logb(feats{f});
    
    % set A and pi also to log value
    logA = Util.log(obj.A);
    logPi = Util.log(obj.Pi);
    
    % compute logdelta and phi
    T = size(feats{f}, 2);
    logdelta = zeros(nStates, T);
    phi = zeros(nStates, T);
    logdelta(:,1) = logPi + logb(:,1);
    for t = 2 : T
        for i = 1 : nStates
            [tmp, index] = max(logdelta(:,t-1) + logA(:,i));
            logdelta(i,t) = tmp + logb(i,t);
            phi(i,t) = index;
        end
    end
    
    % compute optimal states
    states{f} = zeros(1,T);
    [~, states{f}(T)] = max(logdelta(:,T));
    for t = T-1 : -1 : 1
        states{f}(t) = phi(states{f}(t+1), t+1);
    end
end

end