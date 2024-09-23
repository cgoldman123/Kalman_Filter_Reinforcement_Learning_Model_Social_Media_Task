function y = extendnan(x, n) % extend columns with nans
if size(x,1) == 0
    y = nan(0,n);
elseif size(x,2) < n
    y = [x nan(size(x,1), n- size(x,2))];
else
    y = x(:,1:n);
end
end
