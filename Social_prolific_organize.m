function data = Social_prolific_organize(subject, schedule, room_type)
    subj_file = readtable(subject, 'PreserveVariableNames', true);
    start     = max(find(ismember(subj_file.trial_type,'MAIN')))+1;
    subj_file = subj_file(start:end,:);
    subj_file.bandit_num=zeros(height(subj_file),1);
    subj = subj_file(subj_file.event_type==7,:);

    for i=1:height(subj)
        subj(i,:).bandit_num = sum(subj(1:i,:).trial==0) - 1;
    end
    if max(subj.bandit_num)<79
        return
    end
    subj = subj(contains(subj.trial_type, room_type),:);
        
    table_size = [size(subj,1) 14];
    varTypes = ["string", "string", "string", "double", "double", "string", "double", "double", "double", "double", "string", "string", "double", "double"];
    varNames = ["onset", "duration", "trial_type", "trial_number", "bandit_num", "mean_type", "left_reward", "right_reward", "left_mean", "right_mean", "force_pos", "response", "response_time", "points"];

    temp_table = table('Size', table_size, 'VariableTypes', varTypes, 'VariableNames',varNames);
    temp_table.bandit_num = subj.bandit_num;
        
    for i=1:size(room_type,1)
        room=room_type;
        for j=1:size(subj,1)
            temp_table.onset(j)         = 'n/a';
            temp_table.duration(j)      = 'n/a';
            temp_table.trial_type(j)    = extractBefore(subj.trial_type{j}, ['_' room]);
            temp_table.trial_number(j)  = schedule(ismember(schedule.game_number, temp_table.bandit_num),:).trial_num(j);
            temp_table.mean_type(j)     = schedule(ismember(schedule.game_number, temp_table.bandit_num),:).mean_type(j);
            temp_table.left_reward(j)   = schedule(ismember(schedule.game_number, temp_table.bandit_num),:).left_reward(j);
            temp_table.right_reward(j)  = schedule(ismember(schedule.game_number, temp_table.bandit_num),:).right_reward(j);
            temp_table.left_mean(j)     = schedule(ismember(schedule.game_number, temp_table.bandit_num),:).left_mean(j);
            temp_table.right_mean(j)    = schedule(ismember(schedule.game_number, temp_table.bandit_num),:).right_mean(j);
            temp_table.force_pos(j)     = schedule(ismember(schedule.game_number, temp_table.bandit_num),:).force_pos(j);
            temp_table.response(j)      = subj.response{j};
            
            temp_table.response_time(j) = str2double(subj.response_time{j});
            
            points = subj(subj.event_type==7,:).result;
            for pp=1:length(points)
                if isnan(str2double(points{pp}))
                    moved = subj_file(subj_file.event_code==7,:).Var8;
                    points{pp}=num2str(moved(pp));
                end
            end
                temp_table.points(j) = str2double(points{j});
        end
    
        try    
            data{i} = temp_table;
        catch
            fprintf('Made it this far')
        end

    end
    
    data=vertcat(data{:});

end