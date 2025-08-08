function [final_table, started_game] = Berg_local_parse(subject, schedule, room_type, study)
    orgfunc = str2func(['Berg_' study '_organize']);

    started_game = 1;
    data = orgfunc(subject, schedule, room_type);

    if isstring(data.trial_number)
        data.trial_number =str2double(data.trial_number);
    end

    n_games     = length(unique(data.bandit_n));

    filename = split(subject, '\');
    subject_id = split(filename{end}, '-');
    
    final_table = cell(1, n_games);
    for game_i = 1:80
        row = table();
        row.expt_name = 'vertex';
        row.replication_flax = 0;
        row.SubjectID = subject_id(1);  % assign scalar string
        row.order = 0;
        row.age = 22;
        row.gender = 0;
        row.sessionNumber = 1;

        game = data(data.bandit_n == game_i - 1, :);
        if isempty(game)
            continue
        elseif strcmp(room_type, 'Dislike')
            game.left_reward = 100-game.left_reward;
            game.right_reward = 100-game.right_reward;

            game.left_mean(1) = 100-game.left_mean(1);
            game.right_mean(1) = 100-game.right_mean(1);
        end
        row.game = game_i;
        row.gameLength = (size(game, 1));
        row.uc = sum(strcmp(game.forced_choice, 'R'));
        row.m1 = game.left_mean(1);
        row.m2 = game.right_mean(1);
                
        responses = table();
        choices = table();
        reaction_times = table();

        for t = 1:10 %changed from 10
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