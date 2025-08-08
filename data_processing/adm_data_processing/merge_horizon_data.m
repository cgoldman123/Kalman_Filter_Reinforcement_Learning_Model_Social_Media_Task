function [all_data, subject_data_info] = merge_horizon_data(root, ids, groupdata, load_type)        
    all_data = cell(1, 1); 
    
    good_index = [];
    folder = './data';
    
    [all_data{1}, raw_path] = compile_data(folder, ids, groupdata, load_type);  

    if (ismember(size(all_data{1}, 1), 80) && (ismember(sum(all_data{1}.gameLength), 600)))
        good_index = [good_index 1];
    end

    all_data{1}.subjectID = repmat(1, size(all_data{1}, 1), 1); % just set it as 1, doesn't matter much
    

    subject_data_info.id = ids;
    subject_data_info.behavioral_file_path = raw_path;
    subject_data_info.cb = groupdata.hor_task_counterbalance(find(strcmp(groupdata.record_id, ids)));
    subject_data_info.condition = load_type;
    
    
    all_data = all_data(good_index);
    all_data = vertcat(all_data{:}); 
    
    % make reward schedule 
    % Step 1: Initialize an empty table to hold the new structure
    num_games = 80; % There are 80 games
    max_trials = 10; % Max number of trials per game
    reward_schedule = array2table(NaN(num_games, max_trials * 2), ...
        'VariableNames', [strcat('mu1_reward', string(1:max_trials)), ...
                          strcat('mu2_reward', string(1:max_trials))]);
    sub_raw_data = readtable(raw_path, 'FileType', 'text');
    
    % Step 2: Loop through each game_number
    k = 1;
    for game = unique(sub_raw_data.trial_number)'
        % Filter the rows for the current game
        game_rows = sub_raw_data(sub_raw_data.trial_number == game, :);

        % Extract the rewards for mu1 and mu2
        mu1_rewards = game_rows.left_reward;  % Left rewards are for mu1
        mu2_rewards = game_rows.right_reward; % Right rewards are for mu2

        % Determine the number of trials in this game
        num_trials = height(game_rows);

        % Assign rewards to the appropriate columns in the new table
        reward_schedule{k, 1:num_trials} = mu1_rewards';        % mu1 reward columns
        reward_schedule{k, max_trials+1:max_trials+num_trials} = mu2_rewards'; % mu2 reward columns
        k = k+1;
    end
    
    all_data = [all_data, reward_schedule]; 
end