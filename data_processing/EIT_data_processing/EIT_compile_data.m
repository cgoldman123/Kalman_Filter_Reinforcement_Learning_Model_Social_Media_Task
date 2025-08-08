function [final_table, raw_path] = EIT_compile_data(root, subject, groupdata)

  filename = groupdata.filename{find(strcmp(groupdata.subject_id, subject))};
  
  [final_table, raw_path] = fulltable(root, subject, filename);

end

function [fintab, path] = fulltable(rv, sub, filename)
 path = [rv filename];
 data = load(path);
 
 game_struct = data.game;
 % check if only have 79 games (likely all)
 % if only 79 games, key field in row 80 will be empty
 if isempty(game_struct(length(game_struct)).key) 
    % double check key field in row 79
    if ~isempty(game_struct(length(game_struct)-1).key) 
        n_games = length(game_struct)-1;
    end
 else
     n_games = length(game_struct);
 end
    
 fintab = cell(1, n_games);
        
    for game_i = 1:n_games
        row = table();

        row.expt_name = 'vertex';
        row.replication_flag = 0;
        row.subjectID = str2double(sub);
        row.order = 0;
        row.age = 22;
        row.gender = 0;
        
        
        one_game = game_struct(game_i); % stored in one row in game_struct
        
        row.game = game_i;
        row.gameLength = one_game.gameLength;
        row.uc = sum(one_game.nforced == 2); %sum(strcmp(game.force_pos, 'R'));
        row.m1 = one_game.mean(1);%game.left_mean(1);
        row.m2 = one_game.mean(2);%game.right_mean(1);
                
        responses = table();
        choices = table();
        reaction_times = table();
        
        for t = 1:10   
            if t <= row.gameLength 
                choice = one_game.key(t);%convertStringsToChars(game.response(t));
                
                choices.(sprintf('c%d', t)) = one_game.key(t);%strcmp(choice, 'right') + 1;
                responses.(sprintf('r%d', t)) = one_game.reward(t);%game.([choice{1} '_reward'])(t);
                reaction_times.(sprintf('rt%d', t)) = one_game.RT(t);%game.response_time(t);
            else
                responses.(sprintf('r%d', t)) = nan;
                choices.(sprintf('c%d', t)) = nan;
                reaction_times.(sprintf('rt%d', t)) = nan;
            end
        end
        
        for t = 1:4
            reaction_times.(sprintf('rt%d', t)) = nan;
        end        
        
        fintab{game_i} = [row, responses, choices, reaction_times];
    end
   
    fintab = vertcat(fintab{:});
    
    % make reward schedule 
    % Step 1: Initialize an empty table to hold the new structure
    num_games = n_games; % not sure whether will be 79 or 80
    max_trials = 10; % Max number of trials per game
    reward_schedule = array2table(NaN(num_games, max_trials * 2), ...
        'VariableNames', [strcat('mu1_reward', string(1:max_trials)), ...
                          strcat('mu2_reward', string(1:max_trials))]);
    
    
    % Step 2: Loop through each game_number
    
    for game_i = 1:n_games
        
        one_game = game_struct(game_i); % stored in one row in game_struct
        
        % Extract the rewards for mu1 and mu2
        mu1_rewards = one_game.rewards(1,:);  % Left rewards are for mu1
        mu2_rewards = one_game.rewards(2,:); % Right rewards are for mu2

        % Determine the number of trials in this game
        num_trials = one_game.gameLength;

        % Assign rewards to the appropriate columns in the new table
        reward_schedule{game_i, 1:num_trials} = mu1_rewards;        % mu1 reward columns
        reward_schedule{game_i, max_trials+1:max_trials+num_trials} = mu2_rewards; % mu2 reward columns
        
    end
    
    fintab = [fintab, reward_schedule]; 
end