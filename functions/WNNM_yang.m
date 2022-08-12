

function  [X] =  WNNM_yang( Y, C, NSig)
% solve min 1/Nsig|| y-x ||_F^2 + ||x||_w*
% ||x||_w* = sum_i {w_i*sigma_i(x)}
% sigma_i(x) is the singular value of x
    [U,SigmaY,V] = svd(full(Y),'econ');    
    PatNum       = numel(Y)/size(U,2);
    TempC        = C*sqrt(PatNum)*2*NSig^2;
    [SigmaX,svp] = ClosedWNNM(SigmaY,TempC,eps); 
    X =  U(:,1:svp)*diag(SigmaX)*V(:,1:svp)';     


function [SigmaX,svp]=ClosedWNNM(SigmaY,C,oureps)
temp=(SigmaY-oureps).^2 - 4*(C-oureps*SigmaY);
ind=find (temp>0);
svp=length(ind);
SigmaX=max(SigmaY(ind)-oureps+sqrt(temp(ind)),0)/2;
if svp ==0 
    svp = 1;
    SigmaX(1) = SigmaY(1)-oureps;
end


