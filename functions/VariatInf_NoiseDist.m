function model = VariatInf_NoiseDist(model,prior,E)
% Variational inference of NMoG_RPCA. 
E2 = E.^2;
model = nmog_vmax(model, prior, E2);
model = nmog_vexp(model, E2);


function model = nmog_vmax(model,prior,E2)
[m,n] = size(E2);
alpha0 = prior.alpha0;
c0     = prior.c0;
R      = permute(model.R,[1,3,2]);
k      = size(R,3);
for i=1:k
    temp(i,:)  = diag(E2'*R(:,:,i));
end
nk           = reshape(sum(R,1),n,k);
model.alpha  = alpha0 + nk';
model.c      = c0 + nk'/2;
model.d      =  model.eta/model.lambda + 0.5*temp ;
model.eta    = prior.eta0 + k*n*prior.c0;
model.lambda = prior.lambda0 + sum(sum(model.c ./ model.d));

function  model = nmog_vexp(model, E2)
alpha = model.alpha;
c     = model.c;
d     = model.d;
k     = size(c,1);
tau   = c./d;
Elogtau = psi(0, c) - log(d);
Elogpi  = psi(0, alpha) - psi(0, sum(alpha(:)));
for i=1:k
    temp = bsxfun(@times,tau(i,:),E2) ;
    logRho(:,:,i) = (bsxfun(@minus,temp,2*Elogpi(i,:) + Elogtau(i,:) - log(2*pi)))/(-2);
end
logR = bsxfun(@minus,logRho,logsumexp(logRho,3));
R = exp(logR);
model.logR = logR;
model.R = permute(R,[1,3,2]);

