function u_hk = CrankNicolson(A,b,f,u0,ul,ur,T,N)
% we consider the heat equation dt(u) - A*dx^2(u) + b(x,t)*dx(u) = f
%   on [0,1]x[0,T] with u(x,0) = u0(x), u(0,t) = ul(t), u(1,t) = ur(t),
%   and implement Crank-Nicolson based on finite differences in space
% input:
%   A...value of A
%   b...value of b
%   f...function handle on [0,1]x[0,T] of right-hand side of heat equation,
%       should act componentwise on matrices
%   u0...function handle on [0,T] of initial value u0
%   ul...function handle on [0,T] of left value ul
%   ur...function handle on [0,T] of right value ur
%   T...end time point
%   N...number of time points, N-1 is number of time steps
% output:
%   u_hk...matrix of size (N,N) containing Crank-Nicolson approximations at
%       [x_grid,t_grid] = meshgrid(linspace(0,1,N),T*linspace(0,1,N).^grad_t)

x_vec = linspace(0,1,N)';
t_vec = T*linspace(0,1,N)';
h = 1/(N-1);
k = T/(N-1);

u_hk = zeros(N,N);
u_hk(:,1) = ul(t_vec);
u_hk(:,end) = ur(t_vec);
u_hk(1,:) = u0(x_vec);

[x_grid,t_grid] = meshgrid(x_vec(2:(end-1)),t_vec(2:end));
b_all = b(x_grid,t_grid);
f_all = f(x_grid,t_grid);

e = ones(N-2,1);
b_np1 = b(x_vec(2:(end-1)),zeros(N-2,1)); % b^{n+1}
f_np1 = f(x_vec(2:(end-1)),zeros(N-2,1)); % f^{n+1}

q = k/h*0.25;
r = A*k/h^2;
M1 = spdiags([-r*0.5*e (1+r)*e -r*0.5*e],-1:1,N-2,N-2);
M2 = spdiags([r*0.5*e (1-r)*e r*0.5*e],-1:1,N-2,N-2);

for n = 1:(N-1)
    b_n = b_np1; % b^n
    b_np1 = b_all(n,:)';
    f_n = f_np1; % f^n
    f_np1 = f_all(n,:)';
    M1_b = M1 + sparse(2:(N-2),1:(N-3),-q*b_np1(2:end),N-2,N-2) ...
        + sparse(1:(N-3),2:(N-2),q*b_np1(1:(end-1)),N-2,N-2);
    M2_b = M2 + sparse(2:(N-2),1:(N-3),q*b_n(2:end),N-2,N-2) ...
        + sparse(1:(N-3),2:(N-2),-q*b_n(1:(end-1)),N-2,N-2); 
    rhs = M2_b*(u_hk(n,2:(end-1))') + 0.5*k*(f_np1+f_n);
    rhs(1) = rhs(1) + (r*0.5+q*b_np1(1))*u_hk(n+1,1) ...
        + (r*0.5+q*b_n(1))*u_hk(n,1);
    rhs(end) = rhs(end) + (r*0.5-q*b_np1(end))*u_hk(n+1,end) ...
        + (r*0.5-q*b_n(end))*u_hk(n,end);
    u_hk(n+1,2:(end-1)) = M1_b\rhs;
end