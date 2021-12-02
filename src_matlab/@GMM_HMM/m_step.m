function obj = m_step(obj, feats, gamma, i)

nGaussian = obj.nGaussian;
nDim = obj.nDim;
T = size(feats, 2);

% updating
for k = 1 : nGaussian
    mu_tmp = zeros(nDim,1);
    sum_G = sum(gamma(k,:));
    for t = 1 : T
        mu_tmp = mu_tmp + gamma(k,t) * feats(:,t);
    end
    obj.mu{i}(:,k) = mu_tmp / sum_G;
    sig_tmp = zeros(nDim,1);
    for t = 1 : T
        sig_tmp = sig_tmp + gamma(k,t) * (feats(:,t)-obj.mu{i}(:,k)).^2;
    end
    obj.sig{i}(:,k) = sig_tmp / sum_G;
    obj.pik{i}(k) = sum_G / T;
end

end