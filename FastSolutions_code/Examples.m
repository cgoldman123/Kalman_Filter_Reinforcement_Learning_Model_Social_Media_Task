function [F,err,iter] = ...
    Examples(x_rel,t,T,sigma,tol,modID,parameters)
% comments:
%   original PDE: dt F = 1/2 * sigma^2 * dx^2 F + v(x,T-t) * dx F for low(T-t)<=x<=up(T-t), 0<=t<=T
%       with F(x,0) = 0, F(low(T-t),t) = 0, and F(up(T-t),t) = 1
% input:
%   x_rel...matrix with relative space values as entries, x=x_rel*(up(T)-low(T)) 
%       is (pointwise) in [low(T-t),up(T-t)] 
%   t...matrix of size size(x) with time values in [0,T] as entries
%   T...end time point
%   sigma...diffusion coefficient
%	tol...(approximate) error tolerance
%   modID...example model ID:
%       1 = Ornstein-Uhlenbeck
%       2 = hyperbolic urgency
%       3 = linear collapsing bounds
%   parameters...vector with model parameters:
%       if modID = 1: [v0, beta, a]
%       if modID = 2: [v0, v1, tau, a]
%       if modID = 3: [v0, Tinfty, a]
% output:
%   F...F evaluated at (x,t)
%   err...(approximate) error
%   iter...number of iterations

switch modID
    case 1 % Ornstein-Uhlenbeck
        v0 = parameters(1);
        beta = parameters(2);
        a = parameters(3);
        
        % transformed parameters
        %T_bar = 0.5*T/a^2;
        sigma_hat = 0.5*sigma^2*T/a^2;
        v0_hat = beta*T-T*v0/a;
        beta_hat = beta*T;
        
        % transformed drift function
        v_hat = @(x,t) v0_hat - beta_hat*x;
        
        % transformed time and space variable
        x_hat = 1-x_rel;
        t_hat = t/T;
        
    case 2 % Hyperbolic urgency
        v0 = parameters(1);
        v1 = parameters(2);
        tau = parameters(3);
        a = parameters(4);
        
        % transformed parameters
        %T_bar = 0.5*T/a^2;
        sigma_hat = 0.5*sigma^2*T/a^2;
        v0_hat = -v0*T/a;
        v1_hat = -v1*T/a;
        tau_hat = tau/T;
        
        % transformed drift function
        v_hat = @(x,t) v0_hat + v1_hat*(1-t)./((1-t)+tau_hat);
        
        % transformed time and space variable
        x_hat = 1-x_rel;
        t_hat = t/T;
        
    case 3 % Linear collapsing bounds
        v0 = parameters(1);
        Tinfty = parameters(2);
        a = parameters(3);
        
        % transformed parameters
        %T_bar = 0.5*T*Tinfty^2/(a^2*(Tinfty^2-T*Tinfty));
        sigma_hat = 0.5*T*Tinfty*sigma^2/(a^2*(Tinfty-T));
        
        % transformed drift function
        v_hat = @(x,t) T*Tinfty./...
            (a*(Tinfty-T*t)).*(0.5*a/Tinfty*(2*x-1)-v0);
        
        % transformed time and space variable
        x_hat = ((1-x_rel)*Tinfty - 0.5*(T-t))./(Tinfty-T+t);
        t_hat = (t*(Tinfty^2-T*Tinfty))./(T*(t*(Tinfty-T)+(Tinfty-T)^2));
end


[F,err,iter] = FokkerPlanck(x_hat,t_hat,v_hat,sigma_hat,tol);

end