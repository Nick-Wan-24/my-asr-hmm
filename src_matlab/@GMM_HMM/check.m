function norm = check(obj)

norm = 1;
nStates = obj.nStates;

for i = 1 : nStates
    mu = obj.mu{i};
    sig = obj.sig{i};
    if ~isempty(find(isnan(mu), 1)) || ~isempty(find(isnan(sig), 1))
        norm = 0;
        return;
    end
end

end