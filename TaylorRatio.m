function [mu_rT,sig2_rT] = TaylorRatio(mu_N,mu_D,sig2_N,sig2_D,sig2_eta,rho,mu_eta)
%% [mu_rT,sig2_rT] = TaylorRatio(mu_N,mu_D,sig2_N,sig2_D,sig2_eta,rho,mu_eta)
% Mean and variance from the Taylor approximation of the RoG distribution.
%
% Inputs:
%
% - mu_N,mu_D,sig2_N,sig2_D means and variances of the numerator ('N') and
% denominator ('D') of the RoG model
% - sig2_eta variance of the additive noise
% - rho correlation coefficient between z and w
% - mu_eta mean of the additive noise
%
% Outputs:
%
% - mu_rT, sig2_rT the mean and variance of the Taylor approximation to RoG
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

if(~exist('rho','var'))
    rho=0;
end
   
if(~exist('mu_eta','var'))
    mu_eta=0;
end

mu_rT = mu_eta + mu_N./mu_D ;% + mu_N.*sig2_D./(mu_D.^3) - (rho.*((sig2_N.*sig2_D).^.5)./(mu_D.^2)); %* ignore higher-order term 
sig2_rT = sig2_eta + ((mu_N./mu_D).^2) .* ( sig2_N./(mu_N.^2) + sig2_D./(mu_D.^2) - 2*(rho.*((sig2_N.*sig2_D).^.5)./(mu_D.*mu_N)));

end