%% This code illustrates how to use the RoG model, based on the paper: 
%
% "Relating Divisive Normalization to Neuronal Response Variability"
%  Coen-Cagli R. and Solomon, S.S.
%  Journal of Neuroscience 39(37):7344
%
%


%% Check quality of Taylor's approximation
%** Reproduces Fig. 1 of the paper
%** NOTE: the variance is overestimated when w goes too close to 0 (large variance of w), bc the Gaussian approx breaks down

Ntrials = 10^4; % Fig 1b caption wrong, says 10^6
Nconditions = 10^3;
r=NaN(Nconditions,Ntrials); 
mu_N=NaN(Nconditions,1);
mu_D=mu_N;
sig2_N=mu_N;
sig2_D=mu_N;
mu_rT=mu_N;
sig2_rT=mu_N;
sig2_eta=mu_N;

for k=1:Nconditions
    %** specify parameters
    mu_N(k) = rand*100;
    mu_D(k) = .5 + rand*1; 
    alpha_N = 1;
    alpha_D = .001; %** Fig 1b caption wrong, says 0.01
    beta_N = 1 + rand*.5;
    beta_D = beta_N ;
    sig2_N(k) = alpha_N*(mu_N(k)^beta_N);
    sig2_D(k) = alpha_D*(mu_D(k)^beta_D);
    rho = rand/2;
    sig2_eta(k) = .1*mu_N(k)/mu_D(k);
    mu = [mu_N(k) mu_D(k)];
    Sig = [sig2_N(k) rho*sqrt(sig2_N(k)*sig2_D(k)); rho*sqrt(sig2_N(k)*sig2_D(k)) sig2_D(k)];
    %** draw samples
    tmp = repmat(mu,Ntrials,1) + randn(Ntrials,2)*chol(Sig);
    N = tmp(:,1);
    D = tmp(:,2);
    r(k,:) = N./D + randn(Ntrials,1)*sqrt(sig2_eta(k));
    %** Taylor approximation
    [mu_rT(k),sig2_rT(k)] = TaylorRatio(mu_N(k),mu_D(k),sig2_N(k),sig2_D(k),sig2_eta(k),rho);
end
indinf = ~isfinite(r); 
r(indinf) = NaN;

figure; 
subplot(2,3,1); hold on; axis square;
plot(nanmean(r,2),mu_rT,'ok');
m=nanmin([nanmean(r,2) ]);
M=nanmax([nanmean(r,2) ]);
plot([m M],[m M],'--k')
xlabel('Empirical mean')
ylabel('Taylor approximation')
set(gca,'xscale','log','yscale','log')
subplot(2,3,2); hold on; axis square;
plot(nanvar(r,[],2),sig2_rT,'ok');
m=nanmin(nanvar(r,[],2));
M=nanmax(nanvar(r,[],2));
plot([m M],[m M],'--k')
set(gca,'xscale','log','yscale','log')
xlabel('Empirical variance')
ylabel('Taylor approximation')
subplot(2,3,3); hold on; axis square; % Var vs Mean, samples
title('Var vs mean - r');
m=nanmin([nanvar(r,[],2) ]);
M=nanmax([nanvar(r,[],2) ]);
plot(nanmean(r,2),nanvar(r,[],2),'ok');
plot([m M],[m M],'--k')
set(gca,'xscale','log','yscale','log')
subplot(2,3,4); hold on; axis square; % Mean
[h,x] = hist(100*(nanmean(r,2)-mu_rT)./nanmean(r,2),-9.5:1:9.5);
bar(x,h./sum(h));
plot([0 0],[0 .35],'--k');
plot(nanmean(100*(nanmean(r,2)-mu_rT)./nanmean(r,2)),.35,'vk')
xlabel('Count mean (approx.-empirical)/empirical (%)');
ylabel('Proportion of cases');
subplot(2,3,5); hold on; axis square; % Variance
[h,x] = hist(100*(nanvar(r,[],2)-sig2_rT)./nanvar(r,[],2),-9.5:1:9.5);
bar(x,h./sum(h));
plot([0 0],[0 .35],'--k');
plot(nanmean(100*(nanvar(r,[],2)-sig2_rT)./nanvar(r,[],2)),.35,'vk')
xlabel('Count variance (approx.-empirical)/empirical (%)');
ylabel('Proportion of cases');

%% Infer latent state of normalization signal, assuming known parameters
%** Reproduces Fig. 4 of the paper
%*** NOTE: analytical MAP solution *assuming sig2_n=0*

Ntrials = 100;
Nexp = 10000;
Corw = NaN(Nexp,1);
Biasw = NaN(Nexp,1);
DTRUE = NaN(Ntrials,Nexp);
DMAP = NaN(Ntrials,Nexp);
DVAR = NaN(Ntrials,Nexp);
sig2_N = NaN(Nexp,1);
sig2_D = NaN(Nexp,1);
for n=1:Nexp
    %* set parameters
    contrast = 20+rand*30; 
    r_max = 10+rand*90;
    epsilon2 = (15+rand*10)^2;
    rho = 0;
    beta_N = 1.5+rand*.5;
    beta_D = 1.5+rand*.5;
    ctarget2 = 75^2;%* target contrast at which Fano=1
    alpha_N = ((epsilon2 + ctarget2) / (r_max*ctarget2*((r_max*ctarget2)^(beta_N-2)+.01*(epsilon2+ctarget2)^(beta_D-2)))) ; % Enforce Fano=1 at contrast=ctarget
    alpha_D = 10^(rand-.5)*alpha_N;
    mu_N = r_max*contrast^2;
    mu_D = epsilon2 + contrast^2;
    sig2_N(n) = alpha_N*(mu_N^beta_N);
    sig2_D(n) = alpha_D*(mu_D^beta_D);
    mu = [mu_N mu_D];
    Sig = [sig2_N(n) rho*sqrt(sig2_N(n)*sig2_D(n)); rho*sqrt(sig2_N(n)*sig2_D(n)) sig2_D(n)];
    %** draw samples
    tmp = repmat(mu,Ntrials,1) + randn(Ntrials,2)*chol(Sig);
    N = tmp(:,1);
    D = tmp(:,2);
    r = squeeze(N./D);
    %** remove negatives
    indinf = D<=0';%~isfinite(r);
    r(indinf) = NaN;
    N(indinf) = NaN;
    D(indinf) = NaN;
    DTRUE(:,n) = D;
    %** inference
    tmu_z = mu_N*ones(Ntrials,1);
    tmu_w = mu_D*ones(Ntrials,1);
    tsig2_z = sig2_N(n)*ones(Ntrials,1);
    tsig2_w = sig2_D(n)*ones(Ntrials,1);
    tmpwMAP = (r.*tmu_z.*tsig2_w + tmu_w.*tsig2_z) ./ ((r.^2).*tsig2_w + tsig2_z); 
    DMAP(:,n) = tmpwMAP/2 + ((tmpwMAP/2).^2 + tsig2_z.*tsig2_w./((r.^2).*tsig2_w + tsig2_z)).^.5 ; % Posterior MAX
    DVAR(:,n) = ( DMAP(:,n).^(-2) + ((r.^2).*tsig2_w + tsig2_z)./(tsig2_z.*tsig2_w) ).^-1;
    
    indtrials = isfinite(DMAP(:,n).*DTRUE(:,n));
    Corw(n) = (corr(DMAP(indtrials,n),DTRUE(indtrials,n)));
    Biasw(n) = nanmedian(-(DMAP(indtrials,n)-DTRUE(indtrials,n)) ./DTRUE(indtrials,n));%     
end

[varRatio, indsort] = sort(sig2_D./sig2_N);

figure; 
for i=1:3
    subplot(2,3,i); hold on; axis square; axis tight;
    indexp = round((2*i-1)*Nexp/6);
    indexp = indsort(indexp);
    mm=nanmin([DTRUE(:,indexp);DMAP(:,indexp)]);
    MM=nanmax([DTRUE(:,indexp);DMAP(:,indexp)]);
    plot([mm MM],[mm MM],'--k')
    myerrorbar(DTRUE(:,indexp),DMAP(:,indexp),sqrt(DVAR(:,indexp)),'k')
    plot(DTRUE(:,indexp),DMAP(:,indexp),'o')
    xlabel('True denominator');
    ylabel('Inferred denominator');
    set(gca,'xscale','lin','yscale','lin','TickDir','out')
end
subplot(2,3,4);
hold on; axis square;  
[h,x] = hist(Corw,.05:.05:.95);
bar(x,h/sum(h));
plot([.254 .254],[0 max(h/sum(h))],'--k');
xlabel('Corr. Inferred vs true numerator');
ylabel('Number of cases');
set(gca,'xlim',[0 1])
subplot(2,3,5);
hold on; axis square;  
plot(varRatio,Corw(indsort),'.');
plot(varRatio([1 end]),[.254 .254],'--k');
set(gca,'xscale','log','xlim',varRatio([1 end]),'ylim',[0 1])
xlabel('Var ratio');
ylabel('Corr. Inferred vs true numerator');
subplot(2,3,6);
hold on; axis square;  
plot(varRatio,Biasw(indsort),'.');
plot(varRatio([1 end]),[0 0 ],'--k');
xlabel('Var ratio');
ylabel('True - Inferred numerator');
set(gca,'xscale','log','xlim',varRatio([1 end]),'ylim',[-.2 .2])

%% generate synthetic data and fit model 

%* constants
Ntrials = 10000;
Nconditions = 50;
contrast = exp(linspace(log(6),log(100),Nconditions))'; %** NOTE: 'fitRoG' and 'fitRoG_CV' assume contrast on a 0-100 scale, i.e. percent contrast
Nexp = 100;
%*** mean parameters
r_max = 10;
epsilon2 = 25^2;
%*** variance parameters
beta_N = 1.5;
beta_D = 1.3;
alpha_N = 5;
alpha_D = 5;
rho = 0;
%*** spontaneous activity
r_0 = 2;
alpha_eta = 2;
beta_eta = 1;
sig2_eta = alpha_eta*r_0^beta_eta;

%*** true model parameters
otrue = [r_max; epsilon2; alpha_N; beta_N; alpha_D; beta_D; sig2_eta];

o = NaN(7,Nexp);
nLL = NaN(Nexp,1);
for n=1:Nexp
    %* generate data
    r=NaN(Nconditions,Ntrials);
    N=NaN(Nconditions,Ntrials);
    D=NaN(Nconditions,Ntrials);
    mu_z=NaN(Nconditions,1);
    mu_w=mu_z;
    sig2_z=mu_z;
    sig2_w=mu_z;
    for k=1:Nconditions
        %** Evoked mean
        mu_z(k) = r_max*contrast(k)^2;
        mu_w(k) = epsilon2 + contrast(k)^2;

        %*** Evoked variance
        sig2_z(k) = alpha_N*(mu_z(k)^beta_N);
        sig2_w(k) = alpha_D*(mu_w(k)^beta_D);
        
        %** draw samples
        mu = [mu_z(k) mu_w(k)];
        Sig = [sig2_z(k) rho*sqrt(sig2_z(k)*sig2_w(k)); rho*sqrt(sig2_z(k)*sig2_w(k)) sig2_w(k)];
        tmp = repmat(mu,Ntrials,1) + randn(Ntrials,2)*chol(Sig);
        N(k,:) = tmp(:,1); 
        D(k,:) = tmp(:,2); 
        r(k,:) = squeeze(N(k,:)./D(k,:))' + randn(Ntrials,1)*sqrt(sig2_eta) + r_0;
    end
    r(~isfinite(r))=NaN;

    %* fit
    [nLL(n), o(:,n), mu_rT, sig2_rT] = fitRoG(r,contrast,r_0,sig2_eta);      
    
end

figure; 
subplot(1,3,1); hold on; axis square;
plot(contrast,mu_rT,':b','LineWidth',3)
plot(contrast,nanmean(r,2),'ob')
plot(contrast,sig2_rT,':r','LineWidth',3)
plot(contrast,nanvar(r,[],2),'or')
set(gca,'xscale','log','yscale','log')
ylabel('Mean or Var')
xlabel('Contrast %')
subplot(1,3,2); hold on; axis square;
plot(contrast,sig2_rT./mu_rT,':g','LineWidth',3)
plot(contrast,nanvar(r,[],2)./nanmean(r,2),'ok')
set(gca,'xscale','log','yscale','log')
ylabel('Fano')
xlabel('Contrast %')
subplot(1,3,3); hold on; axis square;
[h,x] = hist(nLL,0:.1:1);
bar(x,h,'FaceColor','k','EdgeColor','w');
ylabel('Cases')
xlabel('R-square')

figure; 
for t=1:numel(otrue)
    subplot(1,numel(otrue),t); hold on; axis square;
    hist((abs(o(t,:)-otrue(t)))/otrue(t));
end


%%
