function [all_data, subject_data_info] = Berg_merge(root,ids, files, room_type, study)        
    
    % Note that the ids argument will be as long as the
    % total number of files for all subjects (in the files argument). So there may be 
    % ID repetitions if one ID has multiple behavioral files.  
    
    % This function returns two outputs, all_data and subject_data_info, that
    % will only contain valid subject data. 
   
    % Data is considered valid if it is
    % complete and there are no practice effects (i.e., the subject did not
    % previously start the game). Files are in date order.
    
    
    
    
    
    all_data = cell(1, numel(ids)); 
    good_index = [];
    
    subject_data_info = struct(); 
    
    for i = 1:numel(ids)
        id   = ids{i};
         % only process this ID if haven't previously processed this ID
         % already
        previously_processed_ids = string(ids(1:i-1));
        if ismember(string(id), previously_processed_ids)
            continue;
        end
        file = files(contains(files, id));    
        success=0;
        has_started_a_game = 0;
        for j = 1:numel(file)
            if ~success
                if strcmp(study,'local')
                    filename = file{j};
                    if contains(filename, '_R1-')
                        % Define path
                        file = [root 'rsmith/lab-members/osanchez/wellbeing/social_media/Kalman_Filter_Model_Horizon_Task_Berg/Berg_schedules/horizon_local_modified_v2_9-10-24.csv'];
                        opts = detectImportOptions(file);
                        opts = setvaropts(opts, opts.VariableNames, 'Type', 'char');
                        opts.Delimiter = ',';
                        opts = setvaropts(opts, opts.VariableNames, 'QuoteRule', 'keep');
                        
                        % Read table
                        % Read and split
                        schedule_unprocessed = readtable(file, opts);
                        schedule = table();
                        temp = erase(schedule_unprocessed.Var2, '"');
                        parts = split(temp, "_");

                        game_type_num = str2double(parts(:,2)); % game type indicator(5 = horizon 1, h10 = horizon 6)
                        temp_trial_num = erase(schedule_unprocessed.Var1, '"');
                        schedule.trial_num     = str2double(temp_trial_num);
                        
                        schedule.game_number   = str2double(parts(:, 3));
                        schedule.dislike_room  = str2double(parts(:, 1));
                        schedule.forced_choice = parts(:, 4);
                        schedule.game_type = repmat({'h1'}, height(schedule), 1);
                        schedule.game_type(game_type_num == 10) = {'h6'};
                       
                        temp = erase(schedule_unprocessed.Var5, '"');
                        rewards = split(temp, "_"); %left - right reward values
                        
                        schedule.left_reward  = str2double(rewards(:, 1));
                        schedule.right_reward = str2double(rewards(:, 2));

                        clear game_type_num
                        clear temp_trial_num
                        clear temp
                        clear rewards
                    end
                    [all_data{i},started_this_game] = Berg_local_parse(filename, schedule, room_type, study);   
                end
                has_started_a_game = has_started_a_game+started_this_game;
            else
                % continue because we've already found a complete file for
                % this subject (i.e. success==1)
                continue
            end
            
            % this is a good file if it is complete and there are no
            % practice effects
            if((size(all_data{i}, 1) == 40) && (sum(all_data{i}.gameLength) == 300) && (has_started_a_game <= 1))
                good_index = [good_index i];
                good_file = filename;
                success=1;
            end
            
            all_data{i}.subjectID = repmat(i, size(all_data{i}, 1), 1);
            
            subject_data_info.id = id;
            subject_data_info.has_practice_effects = has_started_a_game > 1;
        end
    end
    
    % only take the rows of all_data that are good
    all_data = all_data(good_index);
    all_data = vertcat(all_data{:});    
    subject_data_info.behavioral_file_path = good_file;

    % add in schedule
    is_dislike_type = strcmp(room_type,'Dislike');
    schedule_room_type = schedule(schedule.dislike_room == is_dislike_type, :);
    % Assuming 'schedule_room_type' is your 280x13 table and it has these columns:
    % game_number, trial_num, left_reward (mu1), right_reward (mu2)

    % Step 1: Initialize an empty table to hold the new structure
    num_games = 40; % There are 40 games
    max_trials = 10; % Max number of trials per game
    reward_schedule = array2table(NaN(num_games, max_trials * 2), ...
        'VariableNames', [strcat('mu1_reward', string(1:max_trials)), ...
                          strcat('mu2_reward', string(1:max_trials))]);

    % Step 2: Loop through each game_number
    k = 1;
    for game = unique(schedule_room_type.game_number)'
        % Filter the rows for the current game
        game_rows = schedule_room_type(schedule_room_type.game_number == game, :);

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