
function [X,W,L,C]= WNLRATV2(Noi_H,Ori_H,Rank, ModelPar, param, model, prior)
% min ||W_n .*(Y-X)||_f^2 + lam1*sum_i{||L_i||_w,*} + lam2*||A||_3DTV
% ModelPar: model params
%% initial
if (~isfield(param,'maxiter'))
    maxiter = 1;
else
    maxiter = param.maxiter;
end                   

if (~isfield(param,'tol'))
    tol = 1e-6;
else
    tol = param.tol;
end

if (~isfield(param,'mog_k'))
    mog_k = 3;
else
    mog_k = param.mog_k;
end

if (~isfield(param,'lr_init'))
    lr_init = 'SVD';
else
    lr_init = param.lr_init;
end

if (~isfield(param,'initial_rank'))
    initial_rank = 3;
else
    initial_rank = param.initial_rank;
end

if (~isfield(param,'rankDeRate'))
    rankDeRate = 1;
else
    rankDeRate = param.rankDeRate;
end

if (~isfield(param,'display'))
    display = 1;
else
    display = param.display;
end

%%
% loss = @(Wn,Y,X,lam1,lam2,M1,L,M2,Ws,A,M3,C,rho)(Wn.*(Y-X)+lam1*())
loss = @(Y,X)(sum((Y(:)-X(:)).^2)/numel(Y));
[m,n,b] = size(Noi_H);
sizeD   = size(Noi_H);
Y       = reshape(Noi_H,m*n,b);
ori_Y   = reshape(Ori_H,m*n,b);
[N,B]   = size(Y);
k       = mog_k;

% Initial low-rank component
[u, s, v] = svd(Y, 'econ');
r = initial_rank;
U0 = u*s;
V0 = v;
U_est = U0(:,1:r);
V_est = V0(:,1:r);
X_est = U_est*V_est';

%%%%%%%%% Initial model parameters %%%%%%%%%%%%%%%
E = Y - X_est;
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

for j = 1:20
    model = VariatInf_NoiseDist(model,prior,E);
end


% W_n
W_n = Weight_NMoG(model,sizeD);

% W_s
r0 = 1;
Y_rank1 =  reshape(U0(:,1:r0)*V0(:,1:r0)',m,n,B);
% show3Dimg(Y_rank1)
Y_dx = abs(diff_x(Y_rank1, sizeD));
Y_dy = abs(diff_y(Y_rank1, sizeD));
Y_dz = abs(diff_z(Y_rank1, sizeD));
W_s1 = Y_dx(:)/max(Y_dx(:));
W_s2 = Y_dy(:)/max(Y_dy(:));
W_s3 = Y_dz(:)/max(Y_dz(:));
W_s = [W_s1; W_s2; W_s3];
clear W_s1 W_s2 W_s3 Y_dx Y_dy Y_dz
W_s = 1./(W_s+1e-2);
W_s = W_s / max(W_s);


Average = mean(Y_rank1,3);
[Neighbor_arr, Num_arr, Self_arr] =	NeighborIndex(Average, param);

% 
X = reshape(X_est,m,n,B);
X_dx = abs(diff_x(X, sizeD));
X_dy = abs(diff_y(X, sizeD));
X_dz = abs(diff_z(X, sizeD));
A = [X_dx(:); X_dy(:); X_dz(:)];
C = X;


% multiplier
Rho_max = 1e7;
% Rho3 = Lam(1)*mean(W_n(:));
% Rho2 = 0.8*Rho3;
Rho3 = ModelPar.alpha*mean(W_n(:));
Rho2 = ModelPar.belta*Rho3;
Lam = [1, ModelPar.gamma*Rho2];

M1 = zeros(size(X));  % multiplier for D-X-E
M2 = zeros(size(W_s));  % multiplier for Dx_X-U_x*V_x
M3 = zeros(size(X));  % multiplier for Dy_X-U_y*V_y
% Lam = [Lam(1)*Rho1, Lam(2)*Rho2];

param.nSig = param.nSig*sqrt(Rank/B);
%%
iter = 0;
err(1) = 0;
tic
while iter<maxiter
    iter          = iter + 1;
    
    %% update L
    
    % Low rank component update
    W0  = Weight_NMoG(model,sizeD);
    [vv,~]  = sort(W0(:));
    Wmax  = vv(ceil(0.7*N*B));
    %Wmax  = vv(ceil(0.75*N*B));
    W0 = min(W0,Wmax);
%     Y_hat = W0.^2 .* Noi_H + Rho3/2 *(C-M3/Rho3);
    Y_hat = W0.^2 .* Noi_H + Rho3/2 *C-M3/2;
    Y_hat =  Y_hat ./ (W0.^2 + Rho3/2);
    W  = W0 + sqrt(Rho3/2);
%      W  = sqrt(W0.^2 + Rho3/2);
%     [mean(W0(:).^2), min(W0(:).^2), max(W0(:).^2),Rho3/2 ]
    [U_est,V_est,~] = EfficientMCL2(reshape(Y_hat,m*n,b), reshape(W,m*n,b), U_est,V_est,10,1e-6);
    L = reshape(U_est*reshape(V_est,b,r)',m,n,b);
%     Lo(iter)=  norm(Ori_H(:)-L(:),'fro')/norm(Ori_H(:),'fro')
%      show3Dimg(L)
    U_est = reshape(U_est, m,n, r);
%         show3Dimg(U_est)
%          show3Dimg(reshape(Y_hat,m,n,b))
    %     show3Dimg(reshape(W,m,n,b))
%     temp = reshape(Y_hat,m,n,b);
%     show3Dimg(temp(:,:,1:60))
    Average             =   mean(U_est(:,:,1:initial_rank),3);
    [CurPat, Mat]	=	Cub2Patch_yang( U_est, U_est, Average, param );

    if iter==1 || (mod(iter,2)==0)
        param.patnum = param.patnum - 10;                                          % Lower Noise level, less NL patches
        NL_mat  =  Block_matching(Mat,param, Neighbor_arr, Num_arr, Self_arr);
    end
    % non-local low-rank denoising
    [Spa_EPat, Spa_W]    =  NLPatEstimation_yang( NL_mat, Self_arr,  CurPat, param);
    % reconstruct patches to 3-D image
    [Spa_Img, Spa_Wei]   =  Patch2Cub( Spa_EPat, Spa_W, param.patsize, m, n, r );       % Patch to Cubic
    U_est = reshape( Spa_Img./Spa_Wei,N, r);
    %%
%     U_est = reshape(U_est, m,n, r);
%         show3Dimg(U_est)
    U_est = reshape( U_est,N, r);
    %%
    L = reshape(U_est*reshape(V_est,b,r)',m,n,b);
%     show3Dimg(L)
%     Lo(iter)=  norm(Ori_H(:)-L(:),'fro')/norm(Ori_H(:),'fro')
%     if (mod(iter-1,3)==0)
%      if   iter==1 || mod(iter,8)==0
 if   iter==1 || mod(iter,18)==0
        [u, s, v] = svd(reshape(L,m*n,B), 'econ');
        r0 = 1 + r0;
        Y_rank1 =  reshape(u(:,1:r0)*s(1:r0,1:r0)*v(:,1:r0)',m,n,B);
%         show3Dimg(Y_rank1)
        Y_dx = abs(diff_x(Y_rank1, sizeD));
        Y_dy = abs(diff_y(Y_rank1, sizeD));
        Y_dz = abs(diff_z(Y_rank1, sizeD));
        W_s1 = Y_dx(:)/max(Y_dx(:));
        W_s2 = Y_dy(:)/max(Y_dy(:));
        W_s3 = Y_dz(:)/max(Y_dz(:));
        W_s = [W_s1; W_s2; W_s3];
        clear W_s1 W_s2 W_s3 Y_dx Y_dy Y_dz
        W_s = 1./(W_s+1e-2);
        W_s = W_s / max(W_s);
%         tempA = reshape(W_s,[m,n,3*b]);
%      show3Dimg(tempA(:,:,:))
    end
    
    %% update A
    
    DC            = diff3(C,  sizeD);
    temp_Y        = DC + M2/Rho2;
    tau           = Lam(2)/Rho2 *W_s;
    A             = softthre(temp_Y, tau);
    %     DCo= [ mean(temp_Y(:)), max(temp_Y(:)), min(temp_Y(:))]
    
    %% update C
    temp_Y1       = L + M3/Rho3;
    temp_Y2       = A - M2/Rho2;
    DC_balance    = Rho2/Rho3;
    C             = TVFast_fft(temp_Y1,temp_Y2,sizeD,DC_balance);
%     Co(iter)=  norm(Ori_H(:)-C(:),'fro')/norm(Ori_H(:),'fro')
%         [~,mpsnr,~,~,~,~]  = evaluate_HSI(Ori_H,C)

    %% update X
    X = L;
%     tempA = reshape(A,[m,n,3*b]);
%        show3Dimg(tempA(:,:,:))
%          show3Dimg(L(:,:,1:16))
%            show3Dimg(C(:,:,:))
%          show3Dimg(W_n(:,:,1:16))
    %% update multiplier
    %     M1            = M1 + Rho1*(X-L);
    M2            = M2 + Rho2*(DC-A);
    M3            = M3 + Rho3*(L-C);
    %     Rho1           = min(1.5*Rho1, Rho_max);
    %     Rho2           = min(1.1 *Rho2, Rho_max);
    %     Rho3           = min(1.3*Rho3, Rho_max);
    %     Rho1           = min(1.2*Rho1, Rho_max);
    Rho2           = min(1.001 *Rho2, Rho_max);
    Rho3           = min(1.1*Rho3, Rho_max);
    
    %% update rank
    
    if r < Rank 
        E     = reshape(Noi_H - L, N,B);
        [u, s, v] = svd(E, 'econ');
        r_up  = 1;
        u0 = u(:,1:r_up)*s(1:r_up,1:r_up);
        v = v(:,1:r_up);
        model_rest = VariatInf_NoiseDist(model,prior,E);
        W  = NMoG2Weight(model_rest);
        [vv,~]  = sort(W(:));
        Wmax  = vv(ceil(0.8*N*B));
        W = min(W,Wmax);
        [u,v,~] = EfficientMCL2(E, W, u0,v,10,1e-6);
        U_est  = [U_est, u];
        V_est  = [V_est, v];
        r  = r + r_up;
    end
    
    %% update W_n
    L = reshape(U_est*reshape(V_est,b,r)',m,n,b);
    E     = reshape(Noi_H - L, N,B);
    for ll = 1:5
        model = VariatInf_NoiseDist(model,prior,E);
    end
    %     W_n = Weight_NMoG(model,sizeD);
    %     W_n = min(W_n, Rho_max);
    
    %% loss
    
    Loss(iter) = loss(Ori_H,X);
    err(iter) = loss(Noi_H,X);
    if iter >1
    if abs(err(iter)-err(iter-1))<1e-6*err(iter-1)
        break
        Loss,
        err
    end
    end
end
% figure,plot(Lo(2:end),'LineWidth',2),hold on,
% plot(Co(2:end),'LineWidth',2), hold on,
% plot(Xo(2:end),'LineWidth',2)
% set(gca,'FontSize',20)
% legend('L','C','X')
% xlabel('Iter'),ylabel('MSE')


