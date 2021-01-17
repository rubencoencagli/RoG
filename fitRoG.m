function [nLL, o, mu_rT, sig2_rT] = fitRoG(r,X,mu_eta,sig2_eta,whichFit)
%% [nLL, o, mu_rT, sig2_rT] = fitRoG(r,X,mu_eta,sig2_eta,whichFit)
% Fit Ratio of Gaussians (RoG) model.
%
% Inputs:
%
% - r is the data matrix for one neuron/experiment (stimulus conditions x trials) 
% - X is the vector of stimulus values, e.g. contrast levels.
% size(X,1) == size(r,1)
%**** NOTE: 'fitRoG' and 'fitRoG_CV' assume contrast on a 0-100 scale, i.e. percent contrast ****%
% - mu_eta, sig2_eta are the mean and variance of the spontaneous activity
% - whichFit: optional input
%
% Outputs:
%
% - o the vector of best fit parameters [r_max epsilon2 alpha_z beta_z alpha_w beta_w sig2_n rho]
% - nLL the negative log-likelihood, normalized between a null model (0)
% and oracle model (1)
% - mu_rT, sig2_rT the mean and variance predcition based on the best fit
% parameters
%
%
% Copyright (c) 2020, Ruben Coen Cagli. 
% All rights reserved.
% See the file LICENSE for licensing information.
%
%
% For derivation, see:
% Coen-Cagli, Solomon. "Relating divisive normalization to neuronal response variability.". 
% Journal of Neuroscience 2019
%
%%


if ~exist('whichFit')
    whichFit=1;
end

global SIG2_eta
SIG2_eta = sig2_eta;

%* compute data moments
mu_r = nanmean(r,2);
sig2_r=nanvar(r,[],2);
r_max=max(mu_r);

nLL=NaN;

switch whichFit
    case 1 %RoG model, ML fit
            %* o = [r_max epsilon2 alpha_N beta_N alpha_D beta_D sig2_eta]
            %* initialize and bound
            ostart =    [r_max      15^2    .2 1.8 .2/1  1.8   sig2_eta];
            LB =        [r_max/2    1^2     .1  1   .1/1  1     sig2_eta/10];
            UB =        [r_max*2    100^2    20  2   20/1  2    sig2_eta*10];
        
        options = optimoptions(@fmincon,'Algorithm','sqp','MaxIter',10000,'Display','off','TolX',1e-6,'TolFun',1e-6,'TolCon',1e-3);
        %     [o, nLL] = fmincon(@(oo)negloglik2(oo,X,mu_eta,mu_r,sig2_r,'ContrastResp'),ostart,[],[],[],[],LB,UB,[],options); %*** do not fit external noise level
        [o, nLL] = fmincon(@(oo)negloglik(oo,X,mu_eta,mu_r,sig2_r,'ContrastResp'),ostart,[],[],[],[],LB,UB,[],options); %*** fit external noise level
        %**RCC added 2020.08.23
        if isempty(nLL)
            nLL=NaN;
        end
        %**
        nLLsaturated = negloglik_saturated(sig2_r);
        nLLnull = negloglik_null(r,mu_r,sig2_r);
        nLL = (nLL-nLLnull)./(nLLsaturated-nLLnull);
        
        [mu_N,mu_D,sig2_N,sig2_D] = ContrastResp(o,X);
        est_sig2_eta = o(7);
        [mu_rT,sig2_rT] = TaylorRatio(mu_N,mu_D,sig2_N,sig2_D,est_sig2_eta,0,mu_eta);%*** assume rho=0

    case 2 %Gamma-Poisson or NegBin model, ML fit
        %* o = [r_max epsilon2 sigg]
        %* initialize and bound
        sigg0 = nanmean(sqrt(max(.0001,(sig2_r-mu_r)./mu_r.^2)));
        ostart =    [r_max      15^2    sigg0];
        LB =        [r_max/2    1^2     .01];
        UB =        [r_max*2    100^2   Inf];
        
        options = optimoptions(@fmincon,'Algorithm','sqp','MaxIter',10000,'Display','off','TolX',1e-6,'TolFun',1e-6,'TolCon',1e-3);
        [o, nLL] = fmincon(@(oo)negloglik_NegBin(oo,X,r,'ContrastRespMean'),ostart,[],[],[],[],LB,UB,[],options);
        %**RCC added 2020.08.23
        if isempty(nLL)
            nLL=NaN;
        end
        %**
        nLLsaturated = negloglik_NegBin_saturated(r,mu_r,sig2_r);
        nLLnull = negloglik_NegBin_null(r);
        nLL = (nLL-nLLnull)./(nLLsaturated-nLLnull);
        
        mu_rT = ContrastRespMean(o,X);
        sig2_rT = mu_rT + (o(3).^2).*(mu_rT.^2);
     
end


%% RoG likelihood

function nLL = negloglik_null(r,mu_r,sig2_r)
mu_rT = nanmean(r(:));
sig2_rT=nanvar(r(:));

%* expectation of model *negative* log-likelihood under data distribution | per stimulus condition
T1 = log(sig2_rT);%*.5
T2 = ((mu_rT-mu_r).^2)./sig2_rT;%*.5
T3 = sig2_r./sig2_rT;%*.5
%* mean across stimulus conditions
sumT = 0.5*(T1+T2+T3);
nLL = nanmean(sumT(:));

function [nLL] = negloglik_saturated(sig2_r)
%* expectation of model *negative* log-likelihood under data distribution | per stimulus condition
T1 = log(sig2_r);%*.5
T2 = 0;%*.5
T3 = 1;%*.5
%* mean across stimulus conditions
sumT = 0.5*(T1+T2+T3);
nLL = nanmean(sumT(:));

function [nLL] = negloglik(o,X,mu_eta,mu_r,sig2_r,fun)
%* use specific model in 'fun'
[mu_N,mu_D,sig2_N,sig2_D] = feval(fun,o,X);
sig2_eta=o(7);
%* compute Taylor approx for the moments
%*** assume rho=0
[mu_rT,sig2_rT] = TaylorRatio(mu_N,mu_D,sig2_N,sig2_D,sig2_eta,0,mu_eta);

%* expectation of model *negative* log-likelihood under data distribution | per stimulus condition
T1 = log(sig2_rT);%*.5
T2 = ((mu_rT-mu_r).^2)./sig2_rT;%*.5
T3 = sig2_r./sig2_rT;%*.5
%* mean across stimulus conditions
sumT = 0.5*(T1+T2+T3);
nLL = nanmean(sumT(:));

function [nLL] = negloglik2(o,X,mu_eta,mu_r,sig2_r,fun)
global SIG2_eta %*** do not fit external noise level
%* use specific model in 'fun'
[mu_N,mu_D,sig2_N,sig2_D] = feval(fun,o,X);
%* compute Taylor approx for the moments
[mu_rT,sig2_rT] = TaylorRatio(mu_N,mu_D,sig2_N,sig2_D,SIG2_eta,0,mu_eta);

%* expectation of model *negative* log-likelihood under data distribution | per stimulus condition
T1 = log(sig2_rT);%*.5
T2 = ((mu_rT-mu_r).^2)./sig2_rT;%*.5
T3 = sig2_r./sig2_rT;%*.5
%* mean across stimulus conditions
sumT = 0.5*(T1+T2+T3);
nLL = nanmean(sumT(:));


%% Contrast response function

function [mu_N,mu_D,sig2_N,sig2_D] = ContrastResp(o,X)
%* o = [r_max epsilon2 alpha_z beta_z alpha_w beta_w]
mu_N = o(1)*X.^2;
mu_D = o(2) + X.^2;
sig2_N = o(3)*(mu_N.^o(4));
sig2_D = o(5)*(mu_D.^o(6));

function f = ContrastRespMean(o,X)
%* o = [r_max epsilon2 ...]
mu_N = o(1)*X.^2;
mu_D = o(2) + X.^2;
f = mu_N./mu_D;


%% Gamma-Poisson likelihood - based on Goris et al 2014 Nature Neuroscience

function nll=negloglik_NegBin(o,X,r,fun)
%* use specific model in 'fun'
f = feval(fun,o,X); % predicted mean count
f = repmat(f,1,size(r,2));
f = f(:);
sigg = nanmax(o(3),0.01); % lower bound
rpar = ones(size(f))/sigg^2; % equivalent to r = mu^2/(var-mu)
p = rpar./(f+rpar); % this is matlab's parametrization!!! not mu/(r+mu)
tmp = nbinpdf(r(:),rpar,p);
tmp(tmp==0) = eps;
nll = -nanmean(log(tmp));

function nll=negloglik_NegBin_null(r)
mu = nanmean(r(:));
sig2 = nanvar(r(:));
rpar = (mu.^2)./(sig2-mu);
p = rpar./(mu+rpar); % this is matlab's parametrization!!! not mu/(r+mu)
tmp = nbinpdf(r(:),rpar,p);
tmp(tmp==0) = eps;
nll = -nanmean(log(tmp));

function nll=negloglik_NegBin_saturated(r,mu_r,sig2_r)
rpar = (mu_r.^2)./(sig2_r-mu_r);
p = rpar./(mu_r+rpar); % this is matlab's parametrization!!! not mu/(r+mu)
rpar = repmat(rpar,1,size(r,2));
p = repmat(p,1,size(r,2));

tmp = nbinpdf(r(:),rpar(:),p(:));
tmp(tmp==0) = eps;
nll = -nanmean(log(tmp));
