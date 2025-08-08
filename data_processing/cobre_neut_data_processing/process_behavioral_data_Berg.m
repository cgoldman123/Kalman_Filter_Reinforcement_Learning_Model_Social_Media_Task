function processed_data = process_behavioral_data_Berg(raw_data)

    R = raw_data(:, 13:22); % changed from 13:22
    C = raw_data(:,23:32); % changed from 23:32
    RT = raw_data(:,33:42); %changed from 33:42
    %BANDIT1_SCHEDULE = [data{40:48}]; % left bandit
    %BANDIT2_SCHEDULE = [data{49:57}]; % right bandit
    BANDIT1_SCHEDULE = raw_data(:, 44:53); % left bandit
    BANDIT2_SCHEDULE = raw_data(:, 54:63); % right bandit
    
    subject_list = unique(raw_data{:, 3});    
    subjectID = subject_list;
    
    ind = find(strcmp(raw_data{:, 3}, subjectID));
    
    sub.expt_name   = raw_data{ind(1), 1};
    sub.replication = raw_data{ind(1), 2};
    sub.subjectID   = raw_data{ind(1), 3};
    
    sub.game        = raw_data{ind, 8};
    sub.gameLength  = raw_data{ind, 9};
    sub.uc          = raw_data{ind, 10};
    sub.m1          = raw_data{ind, 11};
    sub.m2          = raw_data{ind, 12};
    
    %sub.game        = raw_data{8}(ind);
    %sub.gameLength  = raw_data{9}(ind);
    %sub.uc          = raw_data{10}(ind);
    %sub.m1          = raw_data{11}(ind);
    %sub.m2          = raw_data{12}(ind);
    % Convert table of mixed types to uniform numeric matrix

    sub.r           = R(ind,:);
    sub.a           = C(ind,:);
    sub.RT          = RT(ind,:);
    sub.bandit1_schedule = BANDIT1_SCHEDULE(ind,:);
    sub.bandit2_schedule = BANDIT2_SCHEDULE(ind,:);



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
        'forced_choice_info_diff', forced_choice_info_diff, 'actions',  sub.a,  'RTs', sub.RT, 'rewards', sub.r, 'bandit1_schedule', sub.bandit1_schedule,...
        'bandit2_schedule', sub.bandit2_schedule);
    
