function logb = compute_logb(obj, feats)

nStates = obj.nStates;
T = size(feats, 2);
nGaussian = obj.nGaussian;
nDim = obj.nDim;

% matrix allocation
logpik = cell(1, nStates);
logbk = cell(1, nStates);
for i = 1 : nStates
    logpik{i} = Util.log(obj.pik{i});
    logbk{i} = zeros(nGaussian, T);
end
logb = zeros(nStates, T);

tmp1 = 0.5 * nDim * log(2*pi);
for i = 1 : nStates
    if nGaussian > 1
        for g = 1 : nGaussian
            mu = obj.mu{i}(:,g);
            sig = obj.sig{i}(:,g);
            tmp2 = 0.5 * sum(log(sig));
            for t = 1 : T
                ot = feats(:,t);
                logbk{i}(g,t) = -tmp1 - tmp2 - 0.5*sum((ot-mu).^2./sig);
            end
        end
        for t = 1 : T
            logb(i,t) = Util.logsumexp(logpik{i}' + logbk{i}(:,t));
        end
    else
        mu = obj.mu{i};
        sig = obj.sig{i};
        tmp2 = 0.5 * sum(log(sig));
        for t = 1 : T
            ot = feats(:,t);
            logb(i,t) = -tmp1 - tmp2 - 0.5*sum((ot-mu).^2./sig);
        end
    end
end

end