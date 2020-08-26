% This function is the one used for hyper-parameter tuning in Constrained
% model
% Inputs: 
%   training data:      X, Y
%   Attributes:         for both type of classes, att_seen & att_unseen
%   Unseen class names: unseenclasses
%   Hyperparameters:    
%                       mu0     -- initial mean
%                       k0      -- kappa_0
%                       k1      -- kappa_1
%                       a0 & b0 -- Inverse gamma parameters
%                       K       -- The # of nearest neighbors of the unseen
%                                  class among seen classes
%   Version of algo:    Constrainted 
%
% Outputs:
%   Class predictive covariances:  Sig_s
%   Class predictive means:        mu_s
%   Class predictive DoF:          v_s
%   Class ids:                     class_id
   

function [Sig_s,mu_s,v_s,class_id] = constrained_tuning(X,Y,att_seen,att_unseen,unseenclasses,K,mu0,k0,k1,a0,b0)
    
% data stats
seenclasses = unique(Y);
num_class   = length(seenclasses)+length(unseenclasses);
[n, d]      = size(X);

% Initialize output variables
Sigma       = zeros(num_class, d); % prior on class covariances
Sig_s       = zeros(num_class,d);
mu_s        = zeros(num_class,d);
v_s         = zeros(num_class,1);

% Start with unseen classes, little abbrv for the sake of simplicity
uy          = unseenclasses;
ncl         = length(uy);

% First unseen classes 
count       = 1; 

for i=1:ncl

    % For each unseen class attribute, calculate the distince to seen
    % class attributes
    tmp    = att_unseen(i,:);
    D      = pdist2(att_seen,tmp);
    [~, s_ind] = sort(D,'ascend');

    %st = std(sortvals);
    %K = sum(sortvals < st+sortvals(1));


    %K      = 2;      % Choose the K nearsest seen class attributes
    in     = false(n,1);

    % K closest classes from seen classes + selected unseen class will 
    % construct components of this meta cluster.
    classes=seenclasses(s_ind(1:K));

    % mark the indices of these classes
    nci    = length(classes);
    for j=1:nci
        in(Y==classes(j)) = 1;
    end

    % Select the corresponding data points
    Yi     = Y(in);
    Xi     = X(in,:);

    % Prior from Unconstrained model
    Sigma(count, :) = 1./gamrnd(a0, b0, [1, d]);

    uyi    = unique(Yi);
    ncpi   = length(uyi);
    % Initialize class sufficient stats
    xkl    = zeros(ncpi,d);     % Component means
    Skl    = zeros(ncpi,d);     % Component scatter matrices
    kap    = zeros(ncpi,1);     % Model specific 
    nkl    = zeros(ncpi,1);     % # points in the component

    % Calculate sufficient stats
    for j=1:ncpi
        in        = Yi==uyi(j);
        nkl(j)    = sum(in);
        kap(j)    = nkl(j)*k1/(nkl(j)+k1);
        Xij       = Xi(in,:);
        xkl(j,:)  = mean(Xij,1);  % comp_mean
        Skl(j, :) = gamrnd(nkl(j)*ones(1, d)-1, Sigma(count, :));   % comp_scatter  
    end

    % Model specific parameters calculated
    sumkap = sum(kap);
    kaps   = (sumkap+k0)*k1/(sumkap+k0+k1);
    sumSkl = sum(Skl);                                              % sum of scatters of the meta cluster
    muk    = (sum(xkl.*(kap*ones(1,d)),1)+k0*mu0)/(sum(kap)+k0);    % meta cluster mean
    vsc    = 2*(sum(nkl)-ncpi+a0);%sum(nkl)-ncpi+m-d+1+a0; %             % predictive degrees of freedom
    Smu    = 0; %gamrnd(m*ones(1, d), Sigma(count, :));


    class_id(count,1) = uy(i);
    v_s(count)        = vsc;
    Sig_s(count, :)   = (b0 + sumSkl + Smu)/(((kaps)*v_s(count))/(kaps+1));      
    mu_s(count,:)     = muk;        

    count             = count+1;  
end

% Second: The same procedure for seen classess
uy         = seenclasses;
ncl        = length(uy);

for i=1:ncl

    in     = Y==uy(i);
    Xi     = X(in,:);
    Sigma(count, :) = 1./gamrnd(a0, b0, [1, d]);

    % Since the attributes are beong to seen classes, for the second
    % tour we need to exclude selected class to select 3 new classes

    % Current class stats
    cur_n  = sum(in);
    cur_S  = gamrnd(cur_n*ones(1, d)-1, Sigma(count, :));       % class cov
    cur_mu = mean(Xi,1);                                        % class mean

    % Calculating and Selecting closest classes 
    tmp    = att_seen(i,:);     
    D      = pdist2(att_seen,tmp);
    [sortvals s_ind] = sort(D,'ascend');

    %st = std(sortvals);
    %K = sum(sortvals < st+sortvals(1));

    % Fixed number of nearest  neighbors
    %K      = 2;
    in     = false(n,1);
    classes= seenclasses(s_ind(2:K+1));
    nci    = length(classes);

    % To make sure there are some classes in the neighborhood
    if nci>0 

        % Mark the selected classes indices
        for j=1:nci
            in(Y==classes(j)) = 1;
        end

        Yi           = Y(in);
        Xi           = X(in,:);
        uyi          = unique(Yi);
        ncpi         = length(uyi);

        % Initialize parameters
        xkl          = zeros(ncpi,d);
        Skl          = zeros(ncpi,d);
        kap          = zeros(ncpi,1);
        nkl          = zeros(ncpi,1);

        for j=1:ncpi
            in       = Yi==uyi(j);
            nkl(j)   = sum(in);
            kap(j)   = nkl(j)*k1/(nkl(j)+k1);
            Xij      = Xi(in,:);
            xkl(j,:) = mean(Xij,1);
            Skl(j,:) = gamrnd(nkl(j)*ones(1, d)-1, Sigma(count, :));   % comp_scatter    
        end

        % Model parameters calculations
        sumkap       = sum(kap);
        kaps         = (sumkap+k0)*k1/(sumkap+k0+k1);
        sumSkl       = sum(Skl);
        muk          = (sum(xkl.*(kap*ones(1,d)),1)+k0*mu0)/(sum(kap)+k0);
        vsc          =  2*(sum(nkl)-ncpi+a0);%sum(nkl)-ncpi+m-d+1+a0;% 
        %Smu          = gamrnd(m*ones(1, d), Sigma(count, :));

        % Predictive class parameter calculations
        v_s(count)        = vsc+2*cur_n; % vsc+cur_n; %
        Smu               =((cur_n*kaps)/(kaps+cur_n))*((cur_mu-muk).^2);
        Sig_s(count,:)    = (b0+sumSkl+cur_S+Smu)/(((cur_n+kaps)*v_s(count))/(cur_n+kaps+1));
        mu_s(count,:)     = (cur_n*cur_mu+kaps*muk)/(cur_n+kaps);
        class_id(count,1) = uy(i);
        count             = count+1;

        % The case when there is no any other seen class
        else
            v_s(count)        = 2*(cur_n+a0); %cur_n+m-d+1+a0;
            mu_s(count,:)     = (cur_n*cur_mu+(k0*k1/(k0+k1))*mu0)/(cur_n+(k0*k1/(k0+k1)));
            Smu               = ((cur_n*(k0*k1/(k0+k1)))/((k0*k1/(k0+k1))+cur_n))*((cur_mu-mu0).^2);
            Sig_s(count,:)    = (b0+cur_S+Smu)/(((cur_n+(k0*k1/(k0+k1)))*v_s(count))/(cur_n+(k0*k1/(k0+k1))+1));
            class_id(count,1) = uy(i);
            count             = count+1;
    end
end

    
end


 