function dens = dWaldRace(rt, response, A, B, t0, v, s)
% dens = dWaldRace(rt,response,A,B,t0,v,s)
% rt            : 1×N RTs
% response      : 1×N winner indices (1…m)
% A, B, t0, v, s: 1×m cell arrays, each cell is 1×N vectors
% 
% Returns:
% dens          : 1×N vector of defective‐PDF values

N = numel(rt);
m = numel(v);
dens = zeros(1,N);

for i = 1:m
  sel = (response == i);
  if ~any(sel), continue; end

  % reorder so accumulator i is "first" in each cell array
  Bc   = [B(i),   B([1:i-1,i+1:end])];
  Ac   = [A(i),   A([1:i-1,i+1:end])];
  t0c  = [t0(i),  t0([1:i-1,i+1:end])];
  vc   = [v(i),   v([1:i-1,i+1:end])];
  sc   = [s(i),   s([1:i-1,i+1:end])];

  % compute the joint density for those trials
  dens(sel) = n1Wald( rt(sel), Bc, Ac, t0c, vc, sc );
end
end