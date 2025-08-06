function processed_data = process_behavioral_data_SM(raw_data)

    R = [data{13:21}]; % changed from 13:22
    C = [data{22:30}]; % changed from 23:32
    RT = [data{31:39}]; %changed from 33:42
    BANDIT1_SCHEDULE = [data{40:48}]; % left bandit
    BANDIT2_SCHEDULE = [data{49:57}]; % right bandit
    subject_list = unique(data{3});
    
    subjectID = subject_list;
    
    ind = find(data{3} == subjectID);
    
    sub.expt_name   = {data{1}{ind}}';
    sub.replication = data{2}(ind(1));
    sub.subjectID   = num2str(data{3}(ind(1)));
    
    sub.game        = data{8}(ind);
    sub.gameLength  = data{9}(ind);
    sub.uc          = data{10}(ind);
    sub.m1          = data{11}(ind);
    sub.m2          = data{12}(ind);
    
    sub.r           = R(ind,:);
    sub.a           = C(ind,:);
    sub.RT          = RT(ind,:);
    sub.bandit1_schedule = BANDIT1_SCHEDULE(ind,:);
    sub.bandit2_schedule = BANDIT2_SCHEDULE(ind,:);



    %% prep data structure 
    num_forced_choices = 4;              
    num_free_choices_big_hor = 5;
    num_games = 40; 
    % game length i.e., horion
    horizon_type = nan(1,   num_games);
    dum = sub.gameLength;
    horizon_type(1,1:size(dum,1)) = dum;
    % information difference
    dum = sub.uc - 2;
    forced_choice_info_diff(1, 1:size(dum,1)) = -dum;


 

    horizon_type(horizon_type==num_forced_choices+1) = 1;
    horizon_type(horizon_type==num_forced_choices+num_free_choices_big_hor) = 2; %used to be 10


    processed_data = struct(...
        'horizon_type', horizon_type, 'num_games',  num_games, ...
        'num_forced_choices',   num_forced_choices, 'num_free_choices_big_hor',   num_free_choices_big_hor,...
        'forced_choice_info_diff', forced_choice_info_diff, 'actions',  sub.a,  'RTs', sub.RT, 'rewards', sub.r, 'bandit1_schedule', sub.bandit1_schedule,...
        'bandit2_schedule', sub.bandit2_schedule, 'settings', MDP.settings, 'result_dir', result_dir);
    
