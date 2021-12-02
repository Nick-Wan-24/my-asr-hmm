function gamma = e_step(obj, feats, i)

nGaussian = obj.nGaussian;
T = size(feats, 2);

% compute pi_k * logN(t|mu_k,sig_k)
logN = zeros(nGaussian, T);
tmp1 = 0.5 * obj.nDim * log(2*pi);
for k = 1 : nGaussian
    mu = obj.mu{i}(:,k);
    sig = obj.sig{i}(:,k);
    tmp2 = 0.5 * sum(log(sig));
    for t = 1 : T
        ot = feats(:,t);
        logN(k,t) = -tmp1 - tmp2 - 0.5*sum((ot-mu).^2./sig);
    end
end
N = exp(logN) .* repmat(obj.pik{i}',1,T);

% compute gamma(t,k|ot,model)
sum_N = sum(N);
gamma = N ./ repmat(sum_N, nGaussian, 1);

end