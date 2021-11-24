classdef GMM_HMM
    properties
        nStates
        nGaussian
        nIter
        nDim
        Pi
        A
        mu
        sig
        pik
    end
    methods
        % Construction
        function obj = GMM_HMM(para)
            obj.nStates = para.number_of_states;
            obj.nGaussian = para.number_of_gaussian;
            obj.nIter = para.number_of_iteration;
            obj.nDim = para.dimension_of_vector;
            obj.Pi = zeros(obj.nStates, 1);
            obj.A = zeros(obj.nStates, obj.nStates + 1);
            obj.mu = cell(1, obj.nStates);
            obj.sig = cell(1, obj.nStates);
            obj.pik = cell(1, obj.nStates);
            for i = 1 : obj.nStates
                obj.mu{i} = zeros(obj.nDim, obj.nGaussian);
                obj.sig{i} = zeros(obj.nDim, obj.nGaussian);
                obj.pik{i} = zeros(1, obj.nGaussian);
            end
        end
        obj = init(obj, feats)
        obj = train(obj, feats)
        states = viterbi(obj, feats)
        logb = compute_logb(obj, feats)
        obj = update(obj, feats, states)
        gamma = e_step(obj, feats_all, i)
        obj = m_step(obj, feats_all, gamma, i)
        normal = check(obj);
        ll = compute_ll(obj, feats)
    end

    methods (Static)

        % joint 10 GMM_HMM to 1 big model for continuous digital recognition
        function model = joint(model_cell)
            % Construct big model
            N = length(model_cell);
            if N < 1
                error('no model is given')
            end
            para.number_of_states = N * model_cell{1}.nStates;
            para.number_of_gaussian = model_cell{1}.nGaussian;
            para.number_of_iteration = model_cell{1}.nIter;
            para.dimension_of_vector = model_cell{1}.nDim;
            model = GMM_HMM(para);

            % joint mu, sig and pik
            for i = 1 : N
                for j = 1 : model_cell{1}.nStates
                    i_tmp = (i-1) * model_cell{1}.nStates;
                    model.mu{i_tmp+j} = model_cell{i}.mu{j};
                    model.sig{i_tmp+j} = model_cell{i}.sig{j};
                    model.pik{i_tmp+j} = model_cell{i}.pik{j};
                end
            end

            % joint Pi
            model.Pi((0:N-1)*model_cell{1}.nStates + 1) = 1 / N;

            % joint A
            model.A = model.A(:,1:model.nStates);
            for i = 1 : N
                A_tmp = model_cell{i}.A(:, 1:model.nStates/N);
                i_tmp = (i-1)*model.nStates/N+1 : i*model.nStates/N;
                model.A(i_tmp,i_tmp) = A_tmp;
                model.A(i_tmp(end),(0:N-1)*model_cell{1}.nStates+1) = ...
                    (1 - model.A(i_tmp(end),i_tmp(end))) / N;
            end
        end

        % turn a state sequence to number result
        function result = decode(states, nStates, nWords)
            T = length(states);
            result = zeros(T, 1);
            nStates_unit = nStates / nWords;
            index = 1;
            result(index) = floor((states(1)-1) / nStates_unit);
            for t = 2 : T
                tmp = floor((states(t)-1) / nStates_unit);
                if tmp ~= result(index) || states(t) < states(t-1)
                    index = index + 1;
                    result(index) = tmp;
                end
            end
            result = result(1 : index);
        end
    end

end