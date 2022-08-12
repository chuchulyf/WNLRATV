
function W = NMoG2Weight(model)
c           = model.c;
d           = model.d;
tau         = c./d;
R           = model.R;
[N, k, B]   = size(R);

tau = reshape(tau,k,B);
W = zeros(N,B);
for j = 1:k
    W = W + 0.5*bsxfun(@times,tau(j,:),squeeze(R(:,j,:)));
end

[v,~]  = sort(W(:));
Wmax  = v(ceil(0.9*N*B));
W = min(W,Wmax);