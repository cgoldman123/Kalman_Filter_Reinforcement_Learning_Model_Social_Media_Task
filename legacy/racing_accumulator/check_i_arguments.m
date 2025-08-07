function argOut = check_i_arguments(arg, nn, n_v, dots)
% argOut = check_i_arguments(arg,nn,n_v,dots)
%   Like check_n1_arguments except non‐dots case yields n_v cells.
varname = inputname(1);
if ~iscell(arg)
  if ~isnumeric(arg) || numel(arg) < 1
    error('%s needs to be a numeric vector of length >= 1!', varname);
  end
  if dots
    C = num2cell(arg);
    argOut = cellfun(@(x) repmat(x,1,nn), C, 'UniformOutput', false);
  else
    % replicate the same numeric vector into each of n_v cells
    argOut = repmat({repmat(arg,1,nn)}, 1, n_v);
  end
else
  if ~dots && numel(arg) ~= n_v
    error('if %s is a cell, its length must equal number of accumulators.', varname);
  end
  argOut = cell(size(arg));
  for i=1:numel(arg)
    if ~isnumeric(arg{i}) || numel(arg{i}) < 1
      error('%s{%d} needs to be numeric vector length>=1!', varname, i);
    end
    argOut{i} = repmat(arg{i},1,nn);
  end
end
end
