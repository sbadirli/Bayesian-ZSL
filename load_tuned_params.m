% This function loads the hyperparameter set from CV for a scpecified dataset.
% Please refer to the tuning  range and other tuning details to the Supp.
% Materials.

function [K, k_0, k_1, m, s, a_0 , b_0] = load_tuned_params(dataset, model_version)
    
    dataset = upper(dataset);
    if strcmp(model_version, 'unconstrained')
        dim  = 500; a_0 = 0; b_0 = 0;
        FLO  = [10, 10, 50*dim, 3, 3];
        SUN  = [0.1, 10, 50*dim, 7, 2];
        CUB  = [1, 10, 50*dim, 3, 3];
        AWA1 = [10, 10, 500*dim, 3, 2];
        AWA2 = [10, 10, 500*dim, 3, 2];
        APY  = [10, 10, 500*dim, 9, 4];
        ImageNet  = [0.1, 10, 50*dim, 7, 2]; % same as SUN dataset
        eval(['data = ', dataset,';']);
        data = num2cell(data);
        [k_0, k_1, m, s, K] = deal(data{:});
    else
        m = 0; s = 0;
        FLO  = [0.01, 0.1, 10, 1, 3];
        SUN  = [1, 10, 100, 10, 2];
        CUB  = [1, 10, 10, 1, 2];
        AWA1 = [1, 10, 10, 1, 2];
        AWA2 = [1, 10, 10, 1, 2];
        APY  = [0.1, 10, 10, 1, 4];
        ImageNet  = [1, 10, 100, 10, 2]; % same as SUN dataset
        
        eval(['data = ', dataset,';']);
        data = num2cell(data);
        [k_0, k_1, a_0, b_0, K] = deal(data{:});
    end
end