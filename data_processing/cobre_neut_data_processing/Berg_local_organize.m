function data = Berg_local_organize(subject, schedule, room_type)

    subj_file = readtable(subject, 'PreserveVariableNames', true);
    subj = subj_file(subj_file.event_code==4,:);
    trial_type_split = split(subj.trial_type, '_');

    subj.bandit_n      = str2double(trial_type_split(:, 3));
    G = findgroups(subj.bandit_n);
    
    subj.trial_type    = schedule.game_type(:);
    subj.dislike_room  = schedule.dislike_room;
    subj.forced_choice = trial_type_split(:, 4);
    subj.left_reward   = schedule.left_reward(:);
    subj.right_reward  = schedule.right_reward(:); 
    subj.left_mean     = NaN(height(subj), 1);
    subj.right_mean    = NaN(height(subj), 1);

    for g = 1:max(G)
        idx = find(G ==g);

        left_trials  = idx(subj.forced_choice(idx) == "L");
        right_trials = idx(subj.forced_choice(idx) == "R"); 
        if ~isempty(left_trials)
            subj.left_mean(idx) = mean(subj.left_reward(left_trials));
        end
        if ~isempty(right_trials)
            subj.right_mean(idx) = mean(subj.right_reward(right_trials));
        end
        subj.trial_number(idx) = 0:(length(idx)-1);
        
    end
   

    if strcmp(room_type, 'Like')
        data = subj(subj.dislike_room==0,:);
    else
        data = subj(subj.dislike_room==1,:);
    end
end