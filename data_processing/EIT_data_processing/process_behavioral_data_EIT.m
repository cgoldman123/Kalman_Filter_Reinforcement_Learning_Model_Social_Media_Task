function processed_data = process_behavioral_data_EIT(raw_data)
    
    var_names = raw_data.Properties.VariableNames;
    reward_cols = ~cellfun('isempty', regexp(var_names, '^r\d+$'));
    rewards = table2array(raw_data(:, reward_cols));

    choice_cols = ~cellfun('isempty', regexp(var_names, '^c\d+$'));
    actions = table2array(raw_data(:, choice_cols));

    rt_cols = ~cellfun('isempty', regexp(var_names, '^rt\d+$'));
    RTs = table2array(raw_data(:, rt_cols));

    bandit1_schedule_cols = ~cellfun('isempty', regexp(var_names, '^mu1_reward\d+$'));
    bandit1_schedule = table2array(raw_data(:, bandit1_schedule_cols)); % left bandit
    
    bandit2_schedule_cols = ~cellfun('isempty', regexp(var_names, '^mu2_reward\d+$'));
    bandit2_schedule = table2array(raw_data(:, bandit2_schedule_cols)); % right bandit
    
    bandit1_mean_cols = ~cellfun('isempty', regexp(var_names, '^m1$'));
    bandit1_mean = table2array(raw_data(:, bandit1_mean_cols)); % left bandit
    
    bandit2_mean_cols = ~cellfun('isempty', regexp(var_names, '^m2$'));
    bandit2_mean = table2array(raw_data(:, bandit2_mean_cols)); % right bandit

    num_forced_choices = 4;              
    num_free_choices_big_hor = sum(choice_cols) - num_forced_choices;
    num_games = height(raw_data); 

    % horizon
    game_length = [raw_data.gameLength];
    horizon_type = game_length;
    horizon_type(horizon_type==num_forced_choices+1) = 1; % horizon is 1 for small horizon
    horizon_type(horizon_type==num_forced_choices+num_free_choices_big_hor) = 2; % horizon is 2 for big horizon

    % information difference
    num_forced_choices_right = [raw_data.uc]; % num forced choices on right
    forced_choice_info_diff = -(num_forced_choices_right - 2); % subtract 2 and multiply by -1 so it's -1 when three forced choices are shown on right, +1 when three are shown on left




    processed_data = struct(...
        'horizon_type', horizon_type, 'num_games',  num_games, ...
        'num_forced_choices',   num_forced_choices, 'num_free_choices_big_hor',   num_free_choices_big_hor,...
        'forced_choice_info_diff', forced_choice_info_diff, 'actions',  actions,  'RTs', RTs, 'rewards', rewards, 'bandit1_schedule', bandit1_schedule,...
        'bandit2_schedule', bandit2_schedule, 'bandit1_mean',bandit1_mean, 'bandit2_mean',bandit2_mean);
end  
