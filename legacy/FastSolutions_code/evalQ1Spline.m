function output = evalQ1Spline(nodes_x,nodes_t,coeffs,x,t)
% input:
%   nodes_x...sorted vector without repetition (typically nodes_x(1) and
%       nodes_x(end) are first and last point, respectively, of considered
%       space-interval)
%   nodes_t...sorted vector without repetition (typically nodes_t(1) and
%       nodes_t(end) are first and last point, respectively, of considered
%       time-interval)
%   coeffs...matrix of size [length(nodes_t), length(nodes_x)]
%   x...matrix with space values in [0,1] as entries
%   t...matrix of size size(x) with time values in [0,1] as entries
% output:
%   output...matrix of size(x_grid) with entries output(i,j)
%       = sum_k sum_ell B_{t,k}(t_grid(i,j)) * coeffs(k,ell) * B_{x,ell}(x_grid(i,j)),
%       where B_{x,ell} and B_{t,k} are the continuous and piecewise linear
%       functions that are 1 at nodes_x(ell) and nodes_t(k),
%       respectively, and zero at all other entries of nodes_x and nodes_t,
%       respectively

output = zeros(size(x));

for i=1:size(x,1)
    for j=1:size(x,2)
        x_loc = x(i,j);
        t_loc = t(i,j);
        idx_x = find(nodes_x<=x_loc,1,'last'); % x in [nodes_x(idx_x),nodes_x(idx_x))
        if idx_x==length(nodes_x)
            idx_x = idx_x-1;
        end
        idx_t = find(nodes_t<=t_loc,1,'last'); % t in [nodes_t(idx_t),nodes_t(idx_t))
        if idx_t==length(nodes_t)
            idx_t = idx_t-1;
        end
        x0 = nodes_x(idx_x);
        x1 = nodes_x(idx_x+1);
        t0 = nodes_t(idx_t);
        t1 = nodes_t(idx_t+1);
        tmp_x0 = (x_loc-x1)/(x0-x1);
        tmp_x1 = (x_loc-x0)/(x1-x0);
        tmp_t0 = (t_loc-t1)/(t0-t1);
        tmp_t1 = (t_loc-t0)/(t1-t0);
        output(i,j) = tmp_x0*(tmp_t0*coeffs(idx_t,idx_x) + tmp_t1*coeffs(idx_t+1,idx_x)) ...
            + tmp_x1*(tmp_t0*coeffs(idx_t,idx_x+1) + tmp_t1*coeffs(idx_t+1,idx_x+1));
    end
end