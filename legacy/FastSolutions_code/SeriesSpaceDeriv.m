function dx_F = SeriesSpaceDeriv(x, t, v, a, tol, Kmin)
% input:
%   x...matrix with space values in [0,a] as entries
%   t...matrix of size size(x) with time values in [0,\infty) as entries
%	a...right boundary
%	v...drift constant
%	tol...error tolerance
%	Kmin...half the minimum number of series terms to compute (default: 1)
% output:
%   dx_F...space derivative dx F of series solution F from
%       [Gondan, Blurton, Kesselmeier, 2014] to the Fokker-Planck equation evaluated at x and t:
%       dt F = 1/2 dx^2 F + v dx F with F(x, 0) = 0, F(0, t) = 1, and F(a, t) = 0
% comments:
%   (dx F)(0,0) is set 0
%   if v or max(t) is large an alternative computation is used to avoid
%   NaNs in the solution

x = x/a;
t = t/(2*a^2);
v_til = 2*a*v;

if nargin < 6
    Kmin = 1;
end

v = 0.5*v_til;
t = 2*t;
dx_F = zeros(size(t));

% Truncation criterion for non-alternating part of the series (S1na, S2na)
tM = max(max(t));
xm = min(min(x));
K = find(.5*sqrt(tM)*(2-normcdf(1/sqrt(tM)*(2*((1:50) - 1) + xm))...
    -normcdf(1/sqrt(tM)*(2*(1:50) - xm))) < tol, 1, 'first');

if isfinite(exp(.5*max(max(t))*v^2)) % use alternative computation if v or Tmax is large
    dx_F = NoInfComp(v, x, t, K, Kmin);
else
    % compute solution separately for large and small t to avoid NaNs
    idx = isfinite(exp(.5*(100 + x.^2)./t)); % assuming we never compute more than 100 terms of the series
    dx_F(idx) = InfComp(v, x(idx), t(idx), K, Kmin);
    dx_F(~idx) = NoInfComp(v, x(~idx), t(~idx), K, Kmin);
end

% dx F is not defined at t = 0; define value as lim_{t \rightarrow 0} dx F(x,t)
if all(t==0)
    dx_F = zeros(size(x));
else
    dx_F(t==0) = 0;
end

dx_F = dx_F/a;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function output = M(x)
output = erfcx(x/sqrt(2)) ./ sqrt(2) * sqrt(pi);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dx_F = NoInfComp(v, x, t, K, Kmin)
dx_F = zeros(size(t));

sqt = sqrt(t);
sqtinv = 1./sqt;

k=0;
while true
    % even terms
    rk = k + x;
    S1a = 2 .* normpdf(rk./sqt) .* v .* M((rk - t.*v) ./ sqt);
    S1na = 2 .* normpdf(rk./sqt) .* sqtinv;
    k = k + 1;
    
    % odd terms
    rk = k + (1-x);
    S2a = 2 .* normpdf(rk./sqt) .* v .* M((rk + t.*v) ./ sqt);
    S2na =  -2 .* normpdf(rk./sqt) .* sqtinv;
    k = k + 1;
    dx_F = dx_F - (S1a+S1na) + (S2a+S2na);
    
    if(k > K && k > Kmin); break; end
end
dx_F = dx_F .* exp(-0.5.*t.*v^2-v.*x); % prefactor
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dx_F = InfComp(v, x, t, K, Kmin)
dx_F = zeros(size(t));

sqt = sqrt(t);
sqtinv = 1./sqt;

k=0;
while true
    % even terms
    rk = k + x;
    S1a = erfc(1/sqrt(2)*(rk - t.*v) ./ sqt) .* v .* exp(-rk*v);
    S1na = 2 .* normpdf(rk./sqt) .* sqtinv .* exp(-0.5.*t.*v^2);
    k = k + 1;
    
    % odd terms
    rk = k + (1-x);
    S2a = erfc(1/sqrt(2)*(rk + t.*v) ./ sqt) .* v .* exp(rk*v);
    S2na =  -2 .* normpdf(rk./sqt) .* sqtinv .* exp(-0.5.*t.*v^2);
    k = k + 1;
    dx_F = dx_F - (S1a+S1na) + (S2a+S2na);
    
    % Truncation is based on err/2 iso. exp(-0.5.*t.*v^2) err/2 because
    % the exponential term may introduce zeros so that the loop does not
    % terminate.
    if(k > K && k > Kmin); break; end
end
dx_F = dx_F .* exp(-v.*x); % prefactor
end
