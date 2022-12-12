function v = optimMCL2(Matrix,W,u)
[d n] = size(Matrix);
TX = W.*Matrix;
U = u*ones(1,n);
U = W.*U;
up = sum(TX.*U);
down = sum(U.*U);
v = up./down;