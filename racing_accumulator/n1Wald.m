function dens = n1Wald(rt, Bc, Ac, t0c, vc, sc)
% dens = n1Wald(rt,Bc,Ac,t0c,vc,sc)
% Computes g1(t)=f1(t)* ∏_{j=2..m} [1−F_j(t)]
% Inputs:
%  rt   : 1×K RTs for the trials where accumulator 1 “won”
%  Bc   : 1×m cell of 1×K thresholds (first cell = accumulator 1)
%  Ac   : 1×m cell of 1×K threshold variabilities
%  t0c  : 1×m cell of 1×K non‐decision times
%  vc   : 1×m cell of 1×K drift rates
%  sc   : 1×m cell of 1×K noise scales
%
K = numel(rt);
m = numel(vc);

% build matrix of decision‐times dt(j,:) = rt – t0c{j}
dt = nan(m, K);
for j = 1:m
  dt(j,:) = rt - t0c{j};
end

% 1) Wald‐PDF of accumulator 1
f1 = dWald( dt(1,:), vc{1}, Bc{1}, Ac{1}, sc{1} );

% 2) Multiply by survivor of each other accumulator
surv = ones(1,K);
for j = 2:m
  surv = surv .* (1 - pWald( dt(j,:), vc{j}, Bc{j}, Ac{j}, sc{j} ));
end

dens = f1 .* surv;
end