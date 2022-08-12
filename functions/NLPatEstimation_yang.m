function [ EPat, W ] = NLPatEstimation_yang( NL_mat, Self_arr,  CurPat, Par)
if nargin<5
    lam=1;
end
EPat = zeros(size(CurPat));
W    = zeros(size(CurPat));
for  i      =  1 : length(Self_arr)                                 % For each keypatch group
    Temp    =   CurPat(:, NL_mat(1:Par.patnum,i));                  % Non-local similar patches to the keypatch
    M_Temp  =   repmat(mean( Temp, 2 ),1,Par.patnum);
    Temp    =   Temp - M_Temp;
    Sigma   =   std(Temp(:));
    xx      =  WNNM_yang(Temp, Par.lambda*Par.c1, Sigma);
%         xx      =  WNNM_yang(Temp, lam*Par.c1, Sigma_arr(Self_arr(i)));
%     xxrank(i) = rank(xx);
    E_Temp 	=   xx + M_Temp; % WNNM Estimation
    EPat(:,NL_mat(1:Par.patnum,i))  = EPat(:,NL_mat(1:Par.patnum,i)) + E_Temp;
    W(:,NL_mat(1:Par.patnum,i))     = W(:,NL_mat(1:Par.patnum,i)) + ones(size(CurPat,1),size(NL_mat(1:Par.patnum,i),1));
end
% Rank  = [mean(xxrank),min(xxrank),max(xxrank)]
end
