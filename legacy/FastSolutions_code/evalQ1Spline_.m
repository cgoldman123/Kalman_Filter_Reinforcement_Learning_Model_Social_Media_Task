function output = evalQ1Spline(nodes_x,nodes_t,coeffs,x_vec,t_vec)
% input:
%   nodes_x...sorted vector without repetition (typically nodes_x(1) and
%       nodes_x(end) are first and last point, respectively, of considered
%       space-interval)
%   nodes_t...sorted vector without repetition (typically nodes_t(1) and
%       nodes_t(end) are first and last point, respectively, of considered
%       time-interval)
%   coeffs...matrix of size [length(nodes_t), length(nodes_x)]
%   x_vec...sorted column vector without repetetion and values in
%       [nodes_x(1),nodes_x(end)]
%   t_vec...sorted column vector without repetition and values in
%       [nodes_t(1),nodes_t(end)]
% output:
%   output...matrix of size [length(t_vec), length(x_vec)] with entries output(i,j)
%       = sum_k sum_ell B_{t,k}(t_vec(i)) * coeffs(k,ell) * B_{x,ell}(x_vec(j)),
%       where B_{x,ell} and B_{t,k} are the continuous and piecewise linear
%       functions that are 1 at nodes_x(ell) and nodes_t(k),
%       respectively, and zero at all other entries of nodes_x and nodes_t,
%       respectively

[Bsplines_x_vec, i_x_vec] = evalBsplines(nodes_x,x_vec);
Bsplines_x_vec = Bsplines_x_vec';
i_x_vec = i_x_vec';
[Bsplines_t_vec, i_t_vec] = evalBsplines(nodes_t,t_vec);
Bsplines_t_vec = Bsplines_t_vec';
i_t_vec = i_t_vec';

output = zeros(length(t_vec),length(x_vec));

for idx_x = 1:length(x_vec)
    for idx_t = 1:length(t_vec)
        coeffs_loc = coeffs(i_t_vec(:,idx_t),i_x_vec(:,idx_x));
        Bsplines_x_vec_loc = Bsplines_x_vec(:,idx_x);
        Bsplines_t_vec_loc = Bsplines_t_vec(:,idx_t);
        output(idx_t, idx_x) = output(idx_t, idx_x) ...
            + Bsplines_t_vec_loc'*coeffs_loc*Bsplines_x_vec_loc;
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [output, i_vec] = evalBsplines(nodes,s_vec)
EPS = 1e-10;

knots = [nodes(1); nodes; nodes(end)];
p = 1;

n = length(nodes) - 1;

output = zeros(length(s_vec), p+1);
i_vec = zeros(length(s_vec), p+1);

idx_s = 1;
stop = 0;
j_start = find(nodes<=s_vec(1),1,'last');
j_start = max(1,j_start-1); % for rounding error in previous line
for j = j_start:n
    i_j = (p+1) + (j-1);
    a = nodes(j);
    b = nodes(j + 1);
    s_vec_loc = [];
    s = s_vec(idx_s);
    while (a<s || nearlyEq(a,s))  && (s<b && ~nearlyEq(s,b)) || (j==n && nearlyEq(s,b))
        if j==n && nearlyEq(s,b)
            s = b - EPS;
        end
        s_vec_loc = [s_vec_loc; s];
        idx_s = idx_s + 1;
        if idx_s <= length(s_vec)
            s = s_vec(idx_s);
        else
            stop = 1;
            break;
        end
    end
    if ~isempty(s_vec_loc)
        tmp = idx_s-length(s_vec_loc);
        i_vec(tmp:(idx_s-1),:) = repmat((i_j-p) : i_j, idx_s-tmp, 1);
        for i = (i_j-p) : i_j
            knots_loc = knots(i : (i+p+1));
            if (knots_loc(2) == knots_loc(3)) || (s_vec_loc(1) < knots_loc(2))
                val = (s_vec_loc - knots_loc(1)) / (knots_loc(2) - knots_loc(1));
            else
                val = (knots_loc(3) - s_vec_loc) / (knots_loc(3) - knots_loc(2));
            end
            output(tmp:(idx_s-1), i-i_j+p+1) = val;
        end
        if stop
            break
        end
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function output = nearlyEq(x,y)
EPS = 1e-13;
if ( ((x==0 || y==0) && max(abs(x),abs(y))<=EPS) ...
        || (abs(x-y)<=EPS*max(abs(x),abs(y))) )
    output = 1;
else
    output = 0;
end
end
