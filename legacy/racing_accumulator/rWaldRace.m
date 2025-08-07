function out = rWaldRace(n, v, B, A, t0, s, gf, return_ttf)
% out = rWaldRace(n,v,B,A,t0,s,gf,return_ttf)
%   Simulates n trials of an m‑accumulator Wald race.
%   v, B, A, t0, s may be numeric vectors (1×m) OR 1×m cell arrays of scalars.
%   gf = guess/failure rate (0 by default)
%   return_ttf = if true, returns the full m×n finishing‐time matrix.

  if nargin<7, gf = 0;       end
  if nargin<8, return_ttf = false; end

  % ——————————————————————————————————————————
  % 1) Unwrap cells (if any) into 1×m numeric vectors
  if iscell(v),  v  = [v{:}];  end
  if iscell(B),  B  = [B{:}];  end
  if iscell(A),  A  = [A{:}];  end
  if iscell(s),  s  = [s{:}];  end
  if iscell(t0), t0 = [t0{:}]; end
  % ——————————————————————————————————————————

  % 2) sanitize negatives
  B = max(B,0);
  A = max(A,0);

  % 3) ensure correct orientations
  v  = v(:);      % m×1
  B  = B(:)';     % 1×m
  A  = A(:)';     % 1×m
  s  = s(:)';     % 1×m
  t0 = t0(:);     % m×1 (allows per‐accumulator t0 if you ever want)

  m = numel(v);   % number of accumulators

  % ——————————————————————————————————————————
  % 4) Draw m*n samples from the Wald
  %    This returns a 1×(m*n) row vector
  samples = rWald(m*n, B, v', A, s);

  % 5) Reshape into m×n finishing‐time matrix
  ttf = reshape(samples, m, n);

  % 6) Add non‐decision times (broadcast t0 to m×n)
  ttf = ttf + repmat(t0, 1, n);
  % ——————————————————————————————————————————

  if return_ttf
    out = ttf;
    return
  end

  % 7) pick winners & build output table
  [rt_vals, resp] = min(ttf, [], 1);
  out = table(rt_vals', resp', 'VariableNames', {'RT','R'});

  % 8) apply guess/failure lapses
  if gf > 0
    gf_idx     = rand(n,1) < gf;
    out.RT(gf_idx) = NaN;
    out.R( gf_idx) = 1;
  end
end
