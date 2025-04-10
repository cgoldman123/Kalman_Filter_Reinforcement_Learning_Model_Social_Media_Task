% Run the three example models
% OU model
[x_rel, t] = meshgrid([0.1, 0.4], [0.2, 0.35, 0.5]);
T = max(t, [], 'all');
sigma = 1;
tol = 1e-4;
modID = 1;
parameters = [1, 0.4, 1.2];
Examples(x_rel, t, T, sigma, tol, modID, parameters)

% Hyperbolic drift
[x_rel, t] = meshgrid([0.1, 0.4], [0.2, 0.35, 0.5]);
T = max(t, [], 'all');
sigma = 1;
tol = 1e-4;
modID = 2;
parameters = [1, 0.4, 0.7, 1.3];
F = zeros(size(t));
for i = 1:3
	F(i,:) = Examples(x_rel(i,:), t(i,:), t(i,1), sigma, tol, modID, parameters);
end

% Linear bounds
[x_rel, t] = meshgrid([0.3, 0.8], [0.2, 0.35, 0.39]);
sigma = 1;
tol = 1e-4;
modID = 3;
parameters = [1.2, 1, 1];
F = zeros(size(t));
for i = 1:3
	F(i,:) = Examples(x_rel(i,:), t(i,:), t(i,1), sigma, tol, modID, parameters);
end


% Carter Tests
[x_rel, t] = meshgrid([0.5], [1, 2, 3, 4, 5, 6]);
% [x_rel, t] = meshgrid([0.5, .2, .6], [0.25, 0.50, 0.75, .9]);
sigma = 1;
tol = 1e-4;
modID = 3;
% V0 (constant drift rate), Tinf (time when boundaries collapse
% completely), a (distance between boundaries at T=0)
parameters = [1, 10, 5];
F = zeros(size(t));
for i = 1:size(t,1)
	F(i,:) = Examples(x_rel(i,:), t(i,:), t(i,1), sigma, tol, modID, parameters);
end
F