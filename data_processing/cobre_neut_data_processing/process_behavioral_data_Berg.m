function processed_data = process_behavioral_data_Berg(raw_data)

    rewards = table2array(raw_data(:, 13:22)); % changed from 13:22
    actions = table2array(raw_data(:,23:32)); % changed from 23:32
    RTs = cellfun(@str2double, string(table2cell(raw_data(:,33:42)))); %changed from 33:42

    bandit1_schedule = table2array(raw_data(:, 44:53)); % left bandit
    bandit2_schedule = table2array(raw_data(:, 54:63)); % right bandit

    
    
    sub.gameLength  = raw_data{1:height(raw_data), 9};
    sub.uc          = raw_data{1:height(raw_data), 10};
    sub.m1          = raw_data{1:height(raw_data), 11};
    sub.m2          = raw_data{1:height(raw_data), 12};
    



    %% prep data structure 
    num_forced_choices = 4;              
    num_free_choices_big_hor = 6;
    num_games = 40; 
    % game length i.e., horion
    horizon_type = nan(num_games, 1);
    dum = sub.gameLength;
    horizon_type(1:size(dum,1)) = dum;
    % information difference
    dum = sub.uc - 2;
    forced_choice_info_diff = nan(num_games, 1);  % preallocate as column
    forced_choice_info_diff(1:length(dum)) = -dum;

 

    horizon_type(horizon_type==num_forced_choices+1) = 1;
    horizon_type(horizon_type==num_forced_choices+num_free_choices_big_hor) = 2; %used to be 10


    processed_data = struct(...
        'horizon_type', horizon_type, 'num_games',  num_games, ...
        'num_forced_choices',   num_forced_choices, 'num_free_choices_big_hor',   num_free_choices_big_hor,...
        'forced_choice_info_diff', forced_choice_info_diff, 'actions',  actions,  'RTs', RTs, 'rewards', rewards, 'bandit1_schedule', bandit1_schedule,...
        'bandit2_schedule', bandit2_schedule);
    
