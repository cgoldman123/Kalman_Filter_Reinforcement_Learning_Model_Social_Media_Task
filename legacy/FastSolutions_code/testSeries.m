a = 1;
T = 2;
T_til = 0.5*T/a^2;
k_max = 1000;
v = 20;
v_til = -2*a*v;

F_plus_0 = @(x,t) (1-exp(2*v*x))./(exp(-2*v*(a-x))-exp(2*v*x)); 
dx_F_plus_0 = @(x,t) (-exp(2*v*x)*2*v .* (exp(-2*v*(a-x))-exp(2*v*x)) ...
    - (1-exp(2*v*x)) .* (exp(-2*v*(a-x))*2*v-exp(2*v*x)*2*v))...
    ./ (exp(-2*v*(a-x))-exp(2*v*x)).^2; 

F_plus_k = @(x,t,k) -2*pi*exp((a-x)*v - 0.5*v^2*t) * k .* sin(pi*k*(a-x)/a) ...
    .* exp(-pi^2*k^2/(2*a^2) * t) ./ (a^2*v^2 + pi^2*k^2);
dx_F_plus_k = @(x,t,k) -2*pi*k*exp(-pi^2*k^2/(2*a^2) * t) ./ (a^2*v^2 + pi^2*k^2) ...
    .* (exp((a-x)*v - 0.5*v^2*t)*(-v) .* sin(pi*k*(a-x)/a) ...
    + exp((a-x)*v - 0.5*v^2*t) .* cos(pi*k*(a-x)/a) * (-pi*k/a));

x_vec = linspace(0,1);
t_vec = T_til*linspace(0,1);
[x_grid,t_grid] = meshgrid(x_vec,t_vec);

F_til = F_plus_0(a*(1-x_grid),2*a^2*t_grid);
dx_F_til = -a*dx_F_plus_0(a*(1-x_grid),2*a^2*t_grid);
for k = 1:k_max
    F_til = F_til + F_plus_k(a*(1-x_grid),2*a^2*t_grid,k);
    dx_F_til = dx_F_til + -a*dx_F_plus_k(a*(1-x_grid),2*a^2*t_grid,k);
end

F_til2 = Series(x_grid*a, t_grid*(2*a^2), v_til/(2*a), a, 1e-8);
dx_F_til2 = SD2(x_grid*a, t_grid*(2*a^2), v_til/(2*a), a, 1e-8);

figure(1)
surf(x_grid,t_grid,min(abs(F_til-F_til2),1));
figure(2)
surf(x_grid,t_grid,min(abs(dx_F_til-dx_F_til2),1));
