function F = pWald(t,v,B,A,s)
% exact Wald‑CDF to match R’s SuppDists version (no across‑trial variability)
% t  : 1×N times
% v,B,A,s : scalars or same‑sized vectors

if nargin<5, s=1; end
t = t(:)';  n = numel(t);

% replicate scalars
v = repmat(v(:)',1,ceil(n/numel(v))); v = v(1:n);
B = repmat(B(:)',1,ceil(n/numel(B))); B = B(1:n);
A = repmat(A(:)',1,ceil(n/numel(A))); A = A(1:n);
s = repmat(s(:)',1,ceil(n/numel(s))); s = s(1:n);

% effective threshold variability
a = A/2;
k = B + a;

% parameters
mu     = k ./ v;
lambda = (k ./ s).^2;

% avoid t<=0
F = zeros(1,n);
pos = t>0;
sqt = sqrt(lambda(pos) ./ t(pos));
add = sqt .* (1 + t(pos)./mu(pos));
subOpp = sqt .* (t(pos)./mu(pos) - 1);

% R’s formula: exp(2*lambda/mu)*Φ(−add) + Φ(subOpp)
F(pos) = exp(2*lambda(pos) ./ mu(pos)) .* normcdf(-add) ...
         + normcdf(subOpp);
end