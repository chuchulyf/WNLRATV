function R = R_initialization(X, k)
n = size(X, 2);
idx = randsample(n,k);
m = X(:,idx);
[~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
[u,~,label] = unique(label);
while k ~= length(u)
    idx = randsample(n,k);
    m = X(:,idx);
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1);
    [u,~,label] = unique(label);
end
R = full(sparse(1:n,label,1,n,k,n));