function argOut = check_n1_arguments(arg, nn, n_v, dots)
% argOut = check_n1_arguments(arg,nn,n_v,dots)
%   Ensures arg is either a numeric vector or a cell of numeric vectors.
%   Replicates values to length nn or to n_v cells if dots==TRUE.
%   Throws errors if sizes don’t match.
varname = inputname(1);
if ~iscell(arg)
  if ~isnumeric(arg) || numel(arg) < 1
    error('%s needs to be a numeric vector of length >= 1!', varname);
  end
  if dots
    % convert each element into its own cell and replicate
    C = num2cell(arg);
    argOut = cellfun(@(x) repmat(x,1,nn), C, 'UniformOutput', false);
  else
    argOut = repmat(arg,1,nn);
  end
else
  if ~dots && numel(arg) ~= n_v
    error('if %s is a cell, its length must equal number of accumulators.', varname);
  end
  % For each cell, validate and replicate
  argOut = cell(size(arg));
  for i=1:numel(arg)
    if ~isnumeric(arg{i}) || numel(arg{i})<1
      error('%s{%d} needs to be numeric vector length>=1!', varname, i);
    end
    argOut{i} = repmat(arg{i},1,nn);
  end
  if ~dots
    % flatten back into one vector
    argOut = [argOut{:}];
  end
end
end