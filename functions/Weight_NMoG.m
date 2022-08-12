function W = Weight_NMoG(model,sizeD)
c           = model.c;
d           = model.d;
tau         = c./d;
R           = model.R;
[k,B]       = size(tau);
W           = 0.5*bsxfun(@times,tau(1,:),squeeze(R(:,1,:)));
if k>1
    for j = 2:k
        W = W + 0.5*bsxfun(@times,tau(j,:),squeeze(R(:,j,:)));
    end
end
W = reshape(W,sizeD);
