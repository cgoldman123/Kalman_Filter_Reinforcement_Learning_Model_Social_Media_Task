function F = Series(x, t, v, a, tol, Kmin)
% input:
%   x...matrix with space values in [0,a] as entries
%   t...matrix of size size(x) with time values in [0,\infty) as entries
%	a...right boundary
%	v...drift constant
%	tol...error tolerance
%	Kmin...half the minimum number of series terms to compute (default: 1)
% output:
%   F...series solution from [Gondan, Blurton, Kesselmeier, 2014] to the
%       Fokker-Planck equation evaluated at x and t:
%       dt F = 1/2 dx^2 F + v dx F with F(x, 0) = 0, F(0, t) = 1, and F(a, t) = 0
% comments:
%   F(0,0) is set 0
%   if v or max(t) is large an alternative computation is used to avoid
%   NaNs in the solution

x = x/a;
if nargin < 6
    Kmin = 1;
end

F = zeros(size(t));

if isfinite(exp(.5*max(max(t))*v^2)) % use alternative computation if v or Tmax is large
    F = NoInfComp(t, v, a, x, tol, Kmin);
else
    % compute solution separately for large and small t to avoid NaNs
    idx = isfinite(exp(.5*(100*a + a^2*x.^2)./t)); % assuming we never compute more than 100 terms of the series
    F(idx) = InfComp(t(idx), v, a, x(idx), tol, Kmin);
    F(~idx) = NoInfComp(t(~idx), v, a, x(~idx), tol, Kmin);
end
F(x==0 & t==0) = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function output = M(x)
output = erfcx(x/sqrt(2)) ./ sqrt(2) * sqrt(pi);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function F = NoInfComp(t, v, a, x, tol, Kmin)
F = zeros(size(t));
sqt = sqrt(t);

k=0;
while true
    % even terms
    rk = k*a + a.*x;
    S1 = normpdf(rk./sqt) .* (M((rk - t.*v) ./ sqt) + ...
        M((rk + t.*v) ./ sqt));
    if(all(all(abs(S1) < tol)) && k > Kmin); break; end
    k = k + 1;
    
    % odd terms
    rk = k*a + a.*(1-x);
    S2 = normpdf(rk./sqt) .* (M((rk - t.*v) ./ sqt) + ...
        M((rk + t.*v) ./ sqt));
    F = F + S1 - S2;
    
    if(all(all(abs(S2) < tol)) && k > Kmin); break; end
    k = k + 1;
end
F = F .* exp(-0.5.*t.*v^2-v*a.*x); %prefactor
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function F = InfComp(t, v, a, x, tol, Kmin)
F = zeros(size(t));
sqt = sqrt(t);

k=0;
if(Kmin < 4); Kmin = 4; end % compensate for missing factor exp(.5*t*v^2) in the denominator of M
while true
    % even terms
    rk = k*a + a.*x;
    
    S1 = 0.5*erfc(1/sqrt(2)*(rk - t.*v) ./ sqt) .* exp(-rk*v)+...
        0.5*erfc(1/sqrt(2)*(rk + t.*v) ./ sqt) .* exp(rk*v);
    
    if(all(all(abs(S1(~isnan(S1))) < tol)) && k > Kmin); break; end
    k = k + 1;
    
    % odd terms
    rk = k*a + a.*(1-x);
    
    S2 = 0.5*erfc(1/sqrt(2)*(rk - t.*v) ./ sqt) .* exp(-rk*v)+...
        0.5*erfc(1/sqrt(2)*(rk + t.*v) ./ sqt) .* exp(rk*v);
    F = F + S1 - S2;
    
    if(all(all(abs(S2(~isnan(S2))) < tol)) && k > Kmin); break; end
    k = k + 1;
end
F = F .* exp(-v*a.*x); % prefactor
F(t==0) = 0; % alternative computation creates NaNs as initial condition
end
