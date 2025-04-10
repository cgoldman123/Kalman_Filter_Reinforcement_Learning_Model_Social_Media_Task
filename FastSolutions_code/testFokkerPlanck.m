profile on
x_vec = linspace(0,1,100)';
t_vec = linspace(0,1,100)';
v_hat = @(x,t) x.^2 + 2*t.^3;
T_bar = 3;
tol = 1e-5;
iter_max = 11;
[x_grid,t_grid] = meshgrid(x_vec,t_vec);

[F_hat,err,iter] = FokkerPlanck(x_grid,t_grid,v_hat,T_bar,tol,iter_max);

b = @(x,t) -v_hat(x,t);
f = @(x,t) zeros(size(x));
u0 = @(x) zeros(size(x));
ul = @(t) ones(size(t));
ur = @(t) zeros(size(t));
T = 1;
N = 10000;
nodes_x = linspace(0,1,N)';
nodes_t = linspace(0,1,N)';
tmp = CrankNicolson(T_bar,b,f,u0,ul,ur,T,N); 
F_hat1 = evalQ1Spline(nodes_x,nodes_t,tmp,x_grid,t_grid);

surf(x_grid,t_grid,abs(F_hat-F_hat1))
xlabel('x')
ylabel('t')
profile viewer
profile off