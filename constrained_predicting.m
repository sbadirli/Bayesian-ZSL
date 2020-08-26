% This function predicts the class labels for constrained model by
% calculating log-likelihood for each class and assigns the test point to
% the ca\lass with max likelihood.
%
% Inputs:
%   Data -- X
%   Class specific parameters -- Sig_s, mu_s, v_s, class_id
% Outputs:
%   labels -- ypred
%   probability matrix -- prob_mat


function [ypred, prob_mat] = constrained_predicting(X,Sig_s,mu_s,v_s,class_id)

[ncl, ~] = size(mu_s);
prob_mat = zeros(size(X, 1), ncl);

% For each class calculate the log-likelihood for all data
tic
for j=1:ncl
    const          = gammaln((v_s(j)+1)/2)-gammaln(v_s(j)/2) - 0.5*(log(v_s(j))+log(pi));
    data           = (X-mu_s(j, :)).^2;
    data           = const-0.5*log(Sig_s(j, :)) - 0.5*(v_s(j)+1)*log(1 + (data./(Sig_s(j, :)*v_s(j))));
    prob_mat(:, j) = sum(data, 2); % log_tpdf(X, mu_s(j, :), Sig_s(j, :), v_s(j)); % 
end
%fprintf('Total elapsed time: %.4f\n', toc);
% extracting max likelihoods and assigning labels
[~, bb]  = max(prob_mat,[],2);
ypred    = class_id(bb);
end







        
        
    
    
     