function [x_hat,v_hat,T_bar,trafo_parameters] = ...
    Trafos(x_rel,T,modID,parameters)


switch modID
    case 1 % Ornstein-Uhlenbeck
        v0 = parameters(1);
        beta = parameters(2);
        a = parameters(3);
        
        % transformed parameters
        T_bar = 0.5*T/a^2;
        v0_hat = beta*T-T*v0/a;
        beta_hat = beta*T;
        
        trafo_parameters = [v0_hat, beta_hat];
        
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
        T_bar = 0.5*T/a^2;
        v0_hat = -v0*T/a;
        v1_hat = -v1*T/a;
        tau_hat = tau/T_bar;
        
        trafo_parameters = [v0_hat, v1_hat, tau_hat];
        
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
        T_bar = 0.5*T*Tinfty^2/(a^2*(Tinfty^2-T*Tinfty));
        
        trafo_parameters = [];
        
        % transformed drift function
        v_hat = @(x,t) T_bar*0.5*a*Tinfty*(T-Tinfty)./...
            (a^2*t*(Tinfty-T_bar)).*(a*(2*x-1)/Tinfty-2*v0);
        
        % transformed time and space variable
        x_hat = ((1-xrel)*Tinfty - 0.5*(T-t))./(Tinfty-T+t);
        t_hat = (t*(Tinfty^2-T*Tinfty))./(T*(t*(Tinfty-T)+(Tinfty-T)^2));
end

end