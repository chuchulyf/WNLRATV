function Wn = Wn_est_initial(Noi_H,sng,param,model,prior)
if (~isfield(param,'initial_rank'))
    initial_rank = 10;
else
    initial_rank = param.initial_rank;
end
[m,n,b] = size(Noi_H);
sizeD   = size(Noi_H);
Y       = reshape(Noi_H,m*n,b);
[N,B]   = size(Y);
k       = param.mog_k;

% Initial low-rank component
[u, s, v] = svd(Y, 'econ');
r = initial_rank;
U = u(:,1:r)*(s(1:r,1:r)).^(0.6);
V = (s(1:r,1:r)).^(0.4)*v(:,1:r)';
V = V';

%%%%%%%%% Initial model parameters %%%%%%%%%%%%%%%
E = Y-U*V';
for i=1:B
    model.R(:,:,i) = R_initialization(E(:,i)', k);
    alpha0 = prior.alpha0;
    c0 = prior.c0;
    nxbar = reshape(E(:,i), 1,N)*model.R(:,:,i);
    nxbar = nxbar';
    nk = sum(model.R(:,:,i),1)';
    model.alpha(:,i) = alpha0+nk;
    model.c(:,i) = c0 + nk/2;
    temp = reshape(E(:,i).^2, 1, N)*model.R(:,:,i);
    model.d      =  model.eta/model.lambda + 0.5*temp ;
end

for j = 1:10
    model = VariatInf_NoiseDist(model,prior,E);
end

% W_n
Wn = Weight_NMoG(model,sizeD);

Var  = var(E(:));
sng - Var

