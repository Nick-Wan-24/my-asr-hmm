function ll = compute_ll(obj, feats)

nStates = obj.nStates;
n_file = length(feats);

ll = zeros(1, n_file);
for f = 1 : n_file
    % compute logb matrix
    logb = obj.compute_logb(feats{f});
    
    % compute likelihood, no need to forward
    if obj.nStates < 2 % for GMM model
        ll(f) = sum(logb);
        continue;
    end
    
    % set A and pi also to log value
    logA = Util.log(obj.A);
    logPi = Util.log(obj.Pi);
    
    % compute logalpha
    T = size(feats{f},2);
    logalpha = zeros(nStates, T);
    logalpha(:,1) = logPi + logb(:,1);
    for t = 2 : T
        for i = 1 : nStates
            tmp = logA(:,i) + logalpha(:,t-1);
            logalpha(i,t) = logb(i,t) + Util.logsumexp(tmp);
        end
    end
    
    % compute likelihood
    ll(f) = Util.logsumexp(logalpha(:,T));
end

end