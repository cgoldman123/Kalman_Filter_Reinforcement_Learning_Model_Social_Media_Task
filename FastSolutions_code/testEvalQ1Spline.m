n_x = 20; 
n_t = 100;

nodes_x = linspace(0,1,n_x+1)';
nodes_t = linspace(0,1,n_t+1)';

x_vec = linspace(0,1,7)';
t_vec = linspace(0,1,13)';
[x_grid,t_grid] = meshgrid(x_vec,t_vec);

coeffs = rand(n_t+1,n_x+1); %coeffs(1,2)=1;
spline = evalQ1Spline(nodes_x,nodes_t,coeffs,x_vec,t_vec);
spline_ = evalQ1Spline_(nodes_x,nodes_t,coeffs,x_grid,t_grid);
disp('p=1, spline')
disp(max(max(abs(spline-spline_))))
