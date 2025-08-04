function x = dWald(t, v, B, A, s)
% Exact Wald PDF to match R/SuppDists
% t : 1×N decision times (already rt - t0)
% v,B,A,s : scalars or same‑sized vectors
if nargin<5, s = 1; end

t = t(:)';  n = numel(t);
% broadcast
v = repmat(v(:)',1,ceil(n/numel(v))); v = v(1:n);
B = repmat(B(:)',1,ceil(n/numel(B))); B = B(1:n);
A = repmat(A(:)',1,ceil(n/numel(A))); A = A(1:n);
s = repmat(s(:)',1,ceil(n/numel(s))); s = s(1:n);

% compute
k      = B + A/2;
mu     = k ./ v;
lambda = (k ./ s).^2;

x      = zeros(1,n);
pos    = (t>0) & (v>0);
tt     = t(pos);
mup    = mu(pos);
lam    = lambda(pos);

% closed‑form PDF
x(pos) = sqrt(lam ./ (2*pi.*tt.^3)) ...
          .* exp(-lam .* (tt - mup).^2 ./ (2 .* mup.^2 .* tt));
end