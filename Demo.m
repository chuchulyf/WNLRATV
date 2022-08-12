% Demo on i.i.d. Gaussian Noise
clear,clc

%% load data
load WDC
Ori_H = imresize(WDC,[200,200]);
[M, N, B] = size(Ori_H);

%% noise simulated
nSig = 25/255;
sigma_noi = nSig;      % for case 1
for b =1:B
    Noi_H(:,:,b) = Ori_H(:,:,b)  + sigma_noi*randn(M,N);
end
Y = reshape(Noi_H,M*N,B);

noise     = reshape(Noi_H - Ori_H, M*N,B);

%%
disp('############## WNLRATV #################')
Sigma_ratio  = std(noise(:));
initial_rank  = 3;
Rank = 6;
ModelPar.alpha = 30;
ModelPar.belta = 1;
ModelPar.gamma = 0.08;
param   = SetParam_NWT(Noi_H, Sigma_ratio);
param.initial_rank = initial_rank;
param.maxiter = 15;
param.patnum        = 200;
param.lambda        = 2e-1;
[prior, model] = InitialPara( param,0,B);
tic
[Re_hsi,W_n,L,C] = WNLRATV2(Noi_H,Ori_H, Rank,ModelPar, param, model, prior);
toc;

