function [final_table, started_game] = Social_local_parse(subject, schedule, room_type, study)
    orgfunc = str2func(['Social_' study '_organize']);
    try
        subj_data = readtable(subject, 'PreserveVariableNames', true);
        if sum(find(ismember(subj_data.trial_type,'MAIN')))
            started_game = 1;
        else
            started_game = 0;
        end
        data = orgfunc(subject, schedule, room_type);
    catch
        started_game = 0;
        final_table = [];
        return
    end

    if isstring(data.bandit_num)
        data.bandit_num=str2double(data.bandit_num);
    end
    n_games = length(unique(data.bandit_num)); 
    
    final_table = cell(1, n_games);
        
    for game_i = 1:80
        row = table();

        row.expt_name = 'vertex';
        row.replication_flag = 0;
        row.subjectID = str2double(subject);
        row.order = 0;
        row.age = 22;
        row.gender = 0;
        row.sessionNumber = 1;
        
        game = data(data.bandit_num == game_i - 1, :);
        if isempty(game)
            continue
        elseif strcmp(room_type, 'Dislike')
            game.left_reward = 100-game.left_reward;
            game.right_reward = 100-game.right_reward;

            game.left_mean(1) = 100-game.left_mean(1);
            game.right_mean(1) = 100-game.right_mean(1);
        end

        row.game = game_i;
        row.gameLength = size(game, 1);
        row.uc = sum(strcmp(game.force_pos, 'R'));
        row.m1 = game.left_mean(1);
        row.m2 = game.right_mean(1);
                
        responses = table();
        choices = table();
        reaction_times = table();
        
        for t = 1:9 %changed from 10
            if t <= row.gameLength 
                choice = convertStringsToChars(cellstr(game.response(t))); %changed format
                if ~contains(game.response(t), {'right', 'left'})
                    if strcmp(game.response(t), 'period')
                        game.response(t) ='right';
                        choice={'right'};
                    elseif strcmp(game.response(t), 'comma')
                        game.response(t) ='left';
                        choice={'left'};
                    end
                end

                choices.(sprintf('c%d', t)) = strcmp(choice, 'right') + 1;
                responses.(sprintf('r%d', t)) = game.([choice{1} '_reward'])(t);
                reaction_times.(sprintf('rt%d', t)) = game.response_time(t);
            else
                responses.(sprintf('r%d', t)) = nan;
                choices.(sprintf('c%d', t)) = nan;
                reaction_times.(sprintf('rt%d', t)) = nan;
            end
        end
        
        for t = 1:4
            reaction_times.(sprintf('rt%d', t)) = nan;
        end        
        
        final_table{game_i} = [row, responses, choices, reaction_times];
    end
   
    final_table = vertcat(final_table{:});
   

end