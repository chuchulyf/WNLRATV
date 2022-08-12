function param = SetParam_NWT(Noi_H,sigma_noise)
[M, N, B] = size(Noi_H);
param.rankDeRate = 0;         % the number of rank reduced in each iteration
param.mog_k = 3;              % the number of component reduced in each band
param.lr_init = 'SVD';
param.maxiter = 20;
param.tol = 1e-4;
param.display = 1;

% patch setting
param.bandnum = ceil(B/50);
param.nSig =  sigma_noise;
param.patsize = 5;  % patch size
patch_num     = param.patsize*param.patsize;
param.SearchWin = 50;
param.step = 4;
param.RankSelection = 0.01;
param.NumIter = 50;  % Iteration to compute U&V
param.lamada        =   0.56;
param.c1          =  1*sqrt(2);                           % Constant num for HSI


% nSig    = sigma_noise*255;
nSig    = sigma_noise;
if B<=50
    param.initial_rank = ceil(B/3);      % initial rank of low rank component
    param.bandnum       = ceil(B/8);  % number of clean band for patch matching
    param.SearchWin     = 50;                                 % Non-local patch searching window
    param.patnum        = 200;   % patch number in each group
    if nSig<=10.1
        param.patsize       =   5;
        param.patnum        =   200;                          % Increase the patch number and iterations could further improve the performance, at the cost of running time.
    elseif nSig <= 30.1
        param.patsize       =   6;
        param.patnum        =   300;
        param.lamada        =   0.56;
    else
        param.patsize       =   7;
        param.patnum        =   300;
        param.lamada        =   0.54;
    end
    %%%%
elseif B<=100
    param.initial_rank = ceil(B/5);      % initial rank of low rank component
    param.bandnum       = ceil(B/10);
    param.lamada        =  0.7;
    param.patsize       =   5;
    param.SearchWin     =   50;                                 % Non-local patch searching window
    param.patnum        =   300;
    if nSig<=10.1
        param.patnum        =   300;                          % Increase the patch number and iterations could further improve the performance, at the cost of running time.
        param.lamada        =   0.56;
    elseif nSig <= 30.1
        param.patsize       =   5;
        param.patnum        =   400;
        param.lamada        =   0.54;
    else
        param.patsize       =   6;
        param.patnum        =   400;
        param.lamada        =   0.53;
    end
    %%%%
elseif B<=200
    param.initial_rank = ceil(B/6);      % initial rank of low rank component
    param.bandnum       = ceil(B/15);
    param.lamada      = 0.7;
    param.patnum      =   300;
    param.SearchWin   =   50;                                 % Non-local patch searching window
    param.step        = 4;
    if nSig<=10.1
        param.patsize       =   5;
        param.patnum        =   300;                          % Increase the patch number and iterations could further improve the performance, at the cost of running time.
        param.lamada        =   0.56;
    elseif nSig <= 30.1
        param.patsize       =   5;
        param.patnum        =   400;
        param.lamada        =   0.56;
    else
        param.patsize       =   6;
        param.patnum        =   400;
        param.lamada        =   0.54;
    end
else
    param.initial_rank = ceil(B/10);      % initial rank of low rank component
    param.patsize       =   5;
    param.patnum        =   400;
    param.lamada        =   0.5;
end


