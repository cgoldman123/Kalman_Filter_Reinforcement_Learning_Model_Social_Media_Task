function out = rWald(n, B, v, A, s)
% out = rWald(n,B,v,A,s)
%   n : number of samples
%   B : response threshold (vector or scalar)
%   v : drift rate (vector or scalar)
%   A : trial‐to‐trial variability in threshold
%   s : scaling of noise (defaults to 1)
%
% Returns a length‐n vector of first‐passage times.
if nargin < 5, s = 1; end

% Replicate any scalar inputs up to length n
v = repmat(v(:)', 1, ceil(n/numel(v))); v = v(1:n);
B = repmat(B(:)', 1, ceil(n/numel(B))); B = B(1:n);
A = repmat(A(:)', 1, ceil(n/numel(A))); A = A(1:n);
s = repmat(s(:)', 1, ceil(n/numel(s))); s = s(1:n);

out = nan(1,n);              % preallocate output
ok  = v > 0;                 % only positive drift rates terminate

% For each valid trial, draw using rwaldt
bs  = B(ok) + rand(1,sum(ok)).*A(ok);  
out(ok)   = rwaldt(sum(ok), bs, v(ok), s(ok));
out(~ok)  = Inf;             % negative drifts never cross

%=====================================================================
    function x = rwaldt(n, k, l, s)
    % Nested helper: fast Wald/Inverse‐Gaussian RNG
    %   k = threshold
    %   l = rate (drift)
    %   s = noise scale
    tiny = 1e-6;                     
    flag = l > tiny;                % switch between Levy & IG
    
    x = nan(1,n);
    
    % If rate is essentially zero, use a Lévy distribution
    if any(~flag)
      % m=0; c = k^2 for Lévy
      c = k(~flag).^2;
      x(~flag) = c ./ norminv(1 - rand(1,sum(~flag))/2).^2;  
    end
    
    % If rate > tiny, use MATLAB’s Inverse Gaussian RNG
    if any(flag)
      mu     = k(flag)./l(flag);                
      lambda = (k(flag)./s(flag)).^2;
      % Statistics Toolbox required
      x(flag) = random('InverseGaussian', mu, lambda, 1, sum(flag));
    end
    
    % Force non‐negative times
    x(x < 0) = max(x);
    end
%=====================================================================
end