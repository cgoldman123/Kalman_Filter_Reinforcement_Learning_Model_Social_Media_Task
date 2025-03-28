function  [gx] = g_KF_SIGMA4_logistic (x, P, u, in)    
    % Pull out parameters from vector P
    for i = 1:numel(in.MDP.observation_params)
        field_name = in.MDP.observation_params{i}; % Extract the field name
        params.(field_name) = P(i); % Assign the value from P
    end
    % Transform parameters back to native space
    if exist("params","var")
        retrans_params = transform_params_SM("untransform", params,in.MDP.observation_params); 
    else
        retrans_params = [];
    end
    % If retrans_params does not contain a parameter that is needed (i.e.,
    % a parameter not fit), add it
    for f = fieldnames(in.MDP.params)'
        if ~isfield(retrans_params, f{1})
            retrans_params.(f{1}) = in.MDP.params.(f{1});
        end
    end


    % u shows the input for the previous trial because learning happens first
    % Thus, u(4)==4 means the fourth outcome was already learned
    if u(3) >= 4
        if u(4)==1 % horizon is 1
            T = 0;
            Y = 1;
        else % horizon is 5
            T = retrans_params.directed_exp;
            Y = retrans_params.random_exp;                    
        end
        
        reward_diff = x(1) - x(2);
        z = .5; % hyperparam controlling steepness of curve
        t = u(3) + 1;
         % % Exponential descent
         
         info_bonus_bandit1 = x(1)*retrans_params.baseline_info_bonus + x(1)*T*(exp(-z*(t-5))-exp(-4*z))/(1-exp(-4*z));
         info_bonus_bandit2 = x(2)*retrans_params.baseline_info_bonus + x(2)*T*(exp(-z*(t-5))-exp(-4*z))/(1-exp(-4*z));

         % Linear descent
         % info_bonus_bandit1 = sigma1(g,t)*baseline_info_bonus + sigma1(g,t)*T*((9 - t)/4);
         % info_bonus_bandit2 = sigma2(g,t)*baseline_info_bonus + sigma2(g,t)*T*((9 - t)/4);

         info_diff = info_bonus_bandit1 - info_bonus_bandit2;
        


        % total uncertainty is variance of both arms
        total_uncertainty = (x(3)^2 + x(4)^2)^.5;
        
         % % Exponential descent
         RE = Y + ((1 - Y) * (1 - exp(-z * (t - 5))) / (1 - exp(-4 * z)));

         % Linear descent
         % RE = Y * ((9 - t)/4);
        
        decision_noise = total_uncertainty*retrans_params.baseline_noise*RE;


        % probability of choosing bandit 2
        p = 1 / (1 + exp(-(reward_diff+info_diff+retrans_params.side_bias)/(decision_noise)));
        gx = 1-p;  
    else
        gx = 0.5; % put in filler value so forced choice has no effect on fitting. 
    end
        

                

