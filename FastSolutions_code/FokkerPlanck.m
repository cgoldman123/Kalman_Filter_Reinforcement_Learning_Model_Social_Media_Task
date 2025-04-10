function [F_hat,err,iter] = ...
    FokkerPlanck(x_hat,t_hat,v_hat,sigma_hat,tol,iter_max)
% comments:
%   original PDE: dt F = 1/2 * dx^2 F + v(x,T-t) * dx F for low(T-t)<=x<=up(T-t), 0<=t<=T
%       with F(x,0) = 0, F(low(T-t),t) = 0, and F(up(T-t),t) = 1
%   transformed PDE: dt F_hat = sigma_hat*dx^2 F_hat + v_hat(x,t)*dx F_hat on [0,1]x[0,1]
%       with F_hat(x,0) = 0, F_hat(0,t) = 1, and F_hat(1,t) = 0
% input:
%   x_hat...matrix with space values in [0,1] as entries
%   t_hat...matrix of size size(x_hat) with time values in [0,1] as entries
%   v_hat...function handle acting pointwise on matrices x and t
%   sigma_hat...transformed diffusion coefficient
%	tol...(approximate) error tolerance
%   iter_max...maximal number of iterations (default:11)
% output:
%   F_hat...F_hat evaluated at (x_hat,t_hat)
%   err...(approximate) error
%   iter...number of iterations

if nargin < 6
    iter_max = 11;
end

% iterative Crank-Nicolson for transformed PDE with zero boundary conditions
n = 1;

o = @(x) zeros(size(x));
b = @(x,t) -v_hat(x,t);
f = @(x,t) SeriesSpaceDeriv(x, sigma_hat*t*2, -b(0,0)/(sigma_hat*2), 1, 1e-16) ...
    .* (b(0,0) - b(x,t));

nodes_fine = linspace(0,1,2*n+1)';
E_h_fine = CrankNicolson(sigma_hat,b,f,o,o,o,1,2*n+1);
err = Inf;
iter = 0;
while (err>tol) && (iter<=iter_max)
    n = 2*n;
    E_h = E_h_fine;
    nodes_fine = linspace(0,1,2*n+1)';
    E_h2fine = zeros(2*n+1,2*n+1);
    E_h2fine(1:2:end,1:2:end) = E_h;
    E_h2fine(2:2:(end-1),1:2:end) = 0.5*(E_h(1:(end-1),:) + E_h(2:end,:));
    E_h2fine(:,2:2:(end-1)) = 0.5*(E_h2fine(:,1:2:(end-2)) + E_h2fine(:,3:2:end));
    E_h_fine = CrankNicolson(sigma_hat,b,f,o,o,o,1,2*n+1);
    diff_fine = E_h_fine - E_h2fine;
    tmp = evalQ1Spline(nodes_fine,nodes_fine,diff_fine,x_hat,t_hat);
    err = max(max(abs(tmp)));
    iter = iter + 1;
end

% solution of transformed PDE with correct boundary conditions
tmp = evalQ1Spline(nodes_fine,nodes_fine,E_h_fine,x_hat,t_hat);
F_hat = tmp + Series(x_hat, sigma_hat*t_hat*2, -b(0,0)/(sigma_hat*2), 1, 1e-16);
