function [final_table, raw_path] = compile_data(root, subject, groupdata, load_type)

 if strcmp(load_type, 'loaded') 
    run_idx = num2str(groupdata(ismember(groupdata.record_id, subject),:).hor_task_counterbalance);
 elseif strcmp(load_type, 'unloaded')
    run_idx = num2str(-groupdata(ismember(groupdata.record_id, subject),:).hor_task_counterbalance+3);
 end

   [final_table, raw_path] = fulltable(root, subject, run_idx);

end

function [fintab, path] = fulltable(rv, sub, run)
 path = [rv '/sub-' sub '/beh/sub-' sub '_ses-t0_task-horizon_run-' run '_events.tsv'];
 data = readtable(path, 'FileType', 'text');

 n_games = max(data.trial_number) + 1;
    
 fintab = cell(1, n_games);
        
    for game_i = 1:n_games
        row = table();

        row.expt_name = 'vertex';
        row.replication_flag = 0;
        row.subjectID = str2double(sub);
        row.order = 0;
        row.age = 22;
        row.gender = 0;
        row.sessionNumber = run;
        
        game = data(data.trial_number == game_i - 1, :);
        
        row.game = game_i;
        row.gameLength = size(game, 1);
        row.uc = sum(strcmp(game.force_pos, 'R'));
        row.m1 = game.left_mean(1);
        row.m2 = game.right_mean(1);
                
        responses = table();
        choices = table();
        reaction_times = table();
        
        for t = 1:10   
            if t <= row.gameLength 
                choice = convertStringsToChars(game.response(t));
                
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
        
        fintab{game_i} = [row, responses, choices, reaction_times];
    end
   
    fintab = vertcat(fintab{:});
end