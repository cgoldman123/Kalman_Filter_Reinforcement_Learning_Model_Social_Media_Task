profile on

u = @(x,t) sin(pi*x) .* cos(pi*t) + (x-0.5).*(1-t);
dt_u = @(x,t) -pi*sin(pi*x) .* sin(pi*t) - (x-0.5);
dx_u = @(x,t) pi*cos(pi*x) .* cos(pi*t) + (1-t);
dx2_u = @(x,t) -pi^2*sin(pi*x) .* cos(pi*t);

u0 = @(x) u(x,0);
ul = @(t) u(0,t);
ur = @(t) u(1,t);

A = 2;
b = @(x,t) sin(x.^3) + cos(t.^2);
f = @(x,t) dt_u(x,t) - A*dx2_u(x,t) + b(x,t).*dx_u(x,t);
T = 3;

N_vec = [];
err_vec = [];
for ell = 2:10
    N = 2^ell;
    x_vec = linspace(0,1,N)';
    t_vec = T*linspace(0,1,N)';
    [x_grid,t_grid] = meshgrid(x_vec,t_vec);
    u_val = u(x_grid,t_grid);
    
    u_hk = CrankNicolson(A,b,f,u0,ul,ur,T,N);
    
    err = u_hk-u_val;
    surf(x_grid,t_grid,err)
    xlabel('x')
    ylabel('t')
    
    N_vec = [N_vec,N];
    err_vec = [err_vec,max(abs(err(:)))];
end

loglog(N_vec,err_vec,'-bx')
hold on
loglog(N_vec,N_vec.^(-2),'--k')
hold off

profile viewer
profile off