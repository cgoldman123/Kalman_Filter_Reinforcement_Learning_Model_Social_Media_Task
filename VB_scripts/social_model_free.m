function ff = social_model_free(root,file, room_type, study,simulated_data)
    %The only difference between the two versions of the schedules is the
    %order of blocks. Since we fit Dislike and Like rooms separately and
    %the same block within each room type always goes first (within a
    %session), we can use the same schedule.
    
     % determine if cb=1 or cb=2
    if strcmp(study,'local')
        if contains(file, '_R1-')
            schedule = readtable([root 'rsmith/wellbeing/tasks/SocialMedia/schedules/sm_distributed_schedule_CB1.csv']);
            cb = 1;
        else
            schedule = readtable([root 'rsmith/wellbeing/tasks/SocialMedia/schedules/sm_distributed_schedule_CB2.csv']);
            cb = 2;
        end
    elseif strcmp(study,'prolific')
        if contains(file, '_CB_')
            schedule = readtable([root 'rsmith/wellbeing/tasks/SocialMedia/schedules/sm_distributed_schedule_CB2.csv']);
            cb = 2;
        else
           schedule = readtable([root 'rsmith/wellbeing/tasks/SocialMedia/schedules/sm_distributed_schedule_CB1.csv']);
            cb = 1;
        end
     end



    orgfunc = str2func(['Social_' study '_organize']);
    subj_data = orgfunc(file, schedule, room_type);

    % for debugging: disp(subj_data);disp(file{:});disp(ses); disp(room_type);
    ses = 999; % filler because session is no longer relevant
    data = parse_table(subj_data, file, ses, 80, room_type);
        
    % Calculate the difference between left and right options
    mean_diff_values = arrayfun(@(x) x.mean(1) - x.mean(2), data);
    for i = 1:numel(data)
        data(i).mean_diff = mean_diff_values(i);
    end


    % If passing in simulated data, replace participants' actual behavior
    % with simulated behavior in "data"
    if ~isempty(fieldnames(simulated_data))
        for game = 1:40
            data(game).key = simulated_data.actions(game, ~isnan(simulated_data.actions(game, :)));
            data(game).reward = simulated_data.rewards(game, ~isnan(simulated_data.rewards(game, :)));
            data(game).accuracy = data(game).correcttot/data(game).nfree;
            data(game).correct = data(game).key == ((data(game).rewards(1,:) < data(game).rewards(2,:)) + 1);
            data(game).correcttot = sum(data(game).correct(5:end));
            data(game).mean_correct = data(game).key == ((data(game).mean(1) < data(game).mean(2)) + 1);

            data(game).left_observed = mean(data(game).reward(data(game).key==1));
            data(game).right_observed = mean(data(game).reward(data(game).key==2));

            left_observed_mean_before_choice5 = mean(data(game).reward(data(game).key(1:4)==1));
            right_observed_mean_before_choice5 = mean(data(game).reward(data(game).key(1:4)==2));

            data(game).choice5_generative_correct = data(game).key(5) == data(game).max_side;
            data(game).choice5_true_correct = data(game).choice5_generative_correct;
            data(game).choice5_observed_correct = data(game).key(5) == ((left_observed_mean_before_choice5 <right_observed_mean_before_choice5) + 1);
            
           left_observed_mean_before_last_choice = mean(data(game).reward(data(game).key(1:end-1)==1));
           right_observed_mean_before_last_choice = mean(data(game).reward(data(game).key(1:end-1)==2));

            data(game).last_generative_correct = data(game).key(end) == data(game).max_side;
            data(game).last_true_correct = data(game).last_generative_correct;
            data(game).last_observed_correct = data(game).key(end) == ((left_observed_mean_before_last_choice < right_observed_mean_before_last_choice) + 1);
            
            
            data(game).got_total = sum(data(game).reward(data(game).key==2)) + sum(data(game).reward(data(game).key==1));
            for rt_index=1:length(data(game).RT)
                data(game).RT(rt_index) = simulated_data.RTs(game,rt_index);
            end
            data(game).RT_choice5 = data(game).RT(5);
            data(game).RT_choiceLast = data(game).RT(end);
            data(game).true_correct_frac = sum(data(game).mean_correct(5:end))/length(data(game).mean_correct(5:end));
        end
    end

    % ff = fit_horizon(data, ses, room_type);
    ff.room_type = room_type;
    ff.counterbalance            = cb;
    % ---------------------------------------------------------------


    
    h5 = data([data.horizon] == 5);
    h1 = data([data.horizon] == 1);
    
    %h1_22 = h1(sum([vertcat(h1.forced_type)'] == 2) == 2);
    h1_13 = h1(sum([vertcat(h1.forced_type)'] == 2) ~= 2);

    %h6_22 = h6(sum([vertcat(h6.forced_type)'] == 2) == 2);
    h5_13 = h5(sum([vertcat(h5.forced_type)'] == 2) ~= 2);
    
    h5_meancor = vertcat(h5.mean_correct);
    h1_meancor = vertcat(h1.mean_correct);
    
    % for Figure1C ------------------------------
    ff.h5_freec1_acc = sum(h5_meancor(:, 5))  / numel(h5);
    ff.h5_freec2_acc = sum(h5_meancor(:, 6))  / numel(h5);
    ff.h5_freec3_acc = sum(h5_meancor(:, 7))  / numel(h5);
    ff.h5_freec4_acc = sum(h5_meancor(:, 8))  / numel(h5);
    ff.h5_freec5_acc = sum(h5_meancor(:, 9))  / numel(h5);
    
    ff.h1_freec1_acc = sum(h1_meancor(:, 5)) / numel(h1);
   

    % Get the probability of choosing the left side for each generative
    % mean difference (left - right)
    mean_diffs = {'-24', '-12', '-08', '-04', '-02', '02', '04', '08', '12', '24'};
    for mean_diff = mean_diffs
        mean_diff_double = str2double(mean_diff);
        mean_diff_char = mean_diff{:};
        if mean_diff_double > 0
            more_or_less = 'more';
        else
            more_or_less = 'less';
        end

        % Do this for H1
        filtered_dat = h1_13([h1_13.mean_diff] == mean_diff_double);  
        ff.(['h1_left_' mean_diff_char(end-1:end) '_' more_or_less '_prob']) = ((filtered_dat(1).key(end) == 1) + (filtered_dat(2).key(end) == 1))/2;
    
        % Do this for H5
        filtered_dat = h5_13([h5_13.mean_diff] == mean_diff_double);  
        ff.(['h5_left_' mean_diff_char(end-1:end) '_' more_or_less '_choice_1_prob']) = ((filtered_dat(1).key(5) == 1) + (filtered_dat(2).key(5) == 1))/2;
        ff.(['h5_left_' mean_diff_char(end-1:end) '_' more_or_less '_choice_2_prob']) = ((filtered_dat(1).key(6) == 1) + (filtered_dat(2).key(6) == 1))/2;
        ff.(['h5_left_' mean_diff_char(end-1:end) '_' more_or_less '_choice_3_prob']) = ((filtered_dat(1).key(7) == 1) + (filtered_dat(2).key(7) == 1))/2;
        ff.(['h5_left_' mean_diff_char(end-1:end) '_' more_or_less '_choice_4_prob']) = ((filtered_dat(1).key(8) == 1) + (filtered_dat(2).key(8) == 1))/2;
        ff.(['h5_left_' mean_diff_char(end-1:end) '_' more_or_less '_choice_5_prob']) = ((filtered_dat(1).key(9) == 1) + (filtered_dat(2).key(9) == 1))/2;
    
    end




    % In H5 games, get the probability of choosing the high info side when
    % it's generative mean is more/less than the low info side
    % For first free choice

    % this code is commented out because now redundant
    % ff.h5_more_info_24_less = pminfo(h5_13, -24);
    % ff.h5_more_info_12_less = pminfo(h5_13, -12);
    % ff.h5_more_info_08_less = pminfo(h5_13, -8);
    % ff.h5_more_info_04_less = pminfo(h5_13, -4);
    % ff.h5_more_info_02_less = pminfo(h5_13, -2);
    % ff.h5_more_info_24_more = pminfo(h5_13, 24);
    % ff.h5_more_info_12_more = pminfo(h5_13, 12);
    % ff.h5_more_info_08_more = pminfo(h5_13, 8);
    % ff.h5_more_info_04_more = pminfo(h5_13, 4);
    % ff.h5_more_info_02_more = pminfo(h5_13, 2);


    
    ff.h1_more_info_24_less_prob = pminfo(h1_13, -24);
    ff.h1_more_info_12_less_prob = pminfo(h1_13, -12);
    ff.h1_more_info_08_less_prob = pminfo(h1_13, -8);
    ff.h1_more_info_04_less_prob = pminfo(h1_13, -4);
    ff.h1_more_info_02_less_prob = pminfo(h1_13, -2);
    ff.h1_more_info_24_more_prob = pminfo(h1_13, 24);
    ff.h1_more_info_12_more_prob = pminfo(h1_13, 12);
    ff.h1_more_info_08_more_prob = pminfo(h1_13, 8);
    ff.h1_more_info_04_more_prob = pminfo(h1_13, 4);
    ff.h1_more_info_02_more_prob = pminfo(h1_13, 2);

    % because there will be exactly two games with each of the above
    % generative mean differences in H1, we can compute the average by
    % averaging these probabilities

    h1_all_high_info_probs = [ff.h1_more_info_24_less_prob, ff.h1_more_info_12_less_prob, ff.h1_more_info_08_less_prob,...
        ff.h1_more_info_04_less_prob, ff.h1_more_info_02_less_prob, ...
          ff.h1_more_info_24_more_prob, ff.h1_more_info_12_more_prob, ff.h1_more_info_08_more_prob,...
          ff.h1_more_info_04_more_prob, ff.h1_more_info_02_more_prob];
    ff.h1_more_info_prob = mean(h1_all_high_info_probs);

    % In H5 games, get the probability of choosing the high info side when
    % it's generative mean is more/less than the low info side
    % For all free choices

    result_struct = struct();
    for game_num = 1:length(h5)
        game = h5(game_num,:);
        choices = game.key;
        for choice_num = 5:9
            % determine if choice was high or low info based on number of
            % observations previously seen for that side
            num_1_choices = sum(choices(1:choice_num-1) == 1);
            num_2_choices = sum(choices(1:choice_num-1) == 2);
            if choices(choice_num) == 1
                made_high_info_choice = num_1_choices < num_2_choices;
                made_low_info_choice = num_1_choices > num_2_choices;
            else
                made_high_info_choice = num_1_choices > num_2_choices;
                made_low_info_choice = num_1_choices < num_2_choices;
            end
            % don't count trials with equal information previously shown for each
            % option
            if made_high_info_choice | made_low_info_choice
                % find generative mean difference between high info and low
                % info option
                if num_1_choices > num_2_choices
                    % Here, choice 2 is high info
                    gen_mean_diff = game.mean(2) - game.mean(1);
                else
                    % Here, choice 1 is high info
                    gen_mean_diff = game.mean(1) - game.mean(2);
                end

                % convert gen mean difference to a char array
                if gen_mean_diff < 0
                    gen_mean_char = sprintf('%d_less', abs(gen_mean_diff));
                else
                    gen_mean_char = sprintf('%d_more', gen_mean_diff);
                end
                % add leading 0 if necessary
                if length(gen_mean_char)==6; gen_mean_char = ['0' gen_mean_char];end;


                % add 0 or 1 to the number of times they have chosen the
                % high info side with a given generative mean
                count_field_name = ['h5_more_info_' gen_mean_char '_choice_' char(string(choice_num - 4)) '_count'];
                total_field_name = ['h5_more_info_' gen_mean_char '_choice_' char(string(choice_num - 4)) '_total'];

                % Initialize fields if they don't exist
                if ~isfield(result_struct, count_field_name)
                    result_struct.(count_field_name) = 0;
                    result_struct.(total_field_name) = 0;
                end

                result_struct.(count_field_name) = result_struct.(count_field_name) + made_high_info_choice;
                result_struct.(total_field_name) = result_struct.(total_field_name) + 1;

            end
        end
    end


    num_high_info_choice_1 = 0;
    num_high_info_choice_2 = 0;
    num_high_info_choice_3 = 0;
    num_high_info_choice_4 = 0;
    num_high_info_choice_5 = 0;
    total_high_or_low_info_choice_1 = 0;
    total_high_or_low_info_choice_2 = 0;
    total_high_or_low_info_choice_3 = 0;
    total_high_or_low_info_choice_4 = 0;
    total_high_or_low_info_choice_5 = 0;

    % take average to get probability of choosing high info side
    fields = fieldnames(result_struct); % Get all field names
    for i = 1:numel(fields)
        field_name = fields{i}; % Get current field name
        if contains(field_name, 'count') % only process field names that contain count
            new_field_name = strrep(field_name, '_count', ''); % Remove "count"
            ff.([new_field_name '_prob']) = result_struct.(field_name)/result_struct.([new_field_name '_total']);
        end


        % collapse across generative mean differences
        if contains(field_name, 'choice_1_count')
            num_high_info_choice_1 = num_high_info_choice_1 + result_struct.(field_name);
        elseif contains(field_name, 'choice_2_count')
            num_high_info_choice_2 = num_high_info_choice_2 + result_struct.(field_name);
        elseif contains(field_name, 'choice_3_count')
            num_high_info_choice_3 = num_high_info_choice_3 + result_struct.(field_name);
        elseif contains(field_name, 'choice_4_count')
            num_high_info_choice_4 = num_high_info_choice_4 + result_struct.(field_name);
        elseif contains(field_name, 'choice_5_count')
            num_high_info_choice_5 = num_high_info_choice_5 + result_struct.(field_name);
        elseif contains(field_name, 'choice_1_total')
            total_high_or_low_info_choice_1 = total_high_or_low_info_choice_1 + result_struct.(field_name);
        elseif contains(field_name, 'choice_2_total')
            total_high_or_low_info_choice_2 = total_high_or_low_info_choice_2 + result_struct.(field_name);
        elseif contains(field_name, 'choice_3_total')
            total_high_or_low_info_choice_3 = total_high_or_low_info_choice_3 + result_struct.(field_name);
        elseif contains(field_name, 'choice_4_total')
            total_high_or_low_info_choice_4 = total_high_or_low_info_choice_4 + result_struct.(field_name);
        elseif contains(field_name, 'choice_5_total')
            total_high_or_low_info_choice_5 = total_high_or_low_info_choice_5 + result_struct.(field_name);
        end
    end

    % Add NaN for cols that don't exist
    gen_mean_diffs = {'24','12','08','04','02'};
    choice_nums = {'1','2','3','4','5'};
    for mean_diff=gen_mean_diffs
        for choice_num=choice_nums
            field = ['h5_more_info_' mean_diff{:} '_more_choice_' choice_num{:} '_prob'];
            if ~isfield(ff,field)
                ff.(field) = NaN;
            end
            field = ['h5_more_info_' mean_diff{:} '_less_choice_' choice_num{:} '_prob'];
            if ~isfield(ff,field)
                ff.(field) = NaN;
            end            
        end
    end



    % get probability of high info choice for each choice number
    ff.h5_more_info_choice_1_prob = num_high_info_choice_1/total_high_or_low_info_choice_1;
    ff.h5_more_info_choice_2_prob = num_high_info_choice_2/total_high_or_low_info_choice_2;
    ff.h5_more_info_choice_3_prob = num_high_info_choice_3/total_high_or_low_info_choice_3;
    ff.h5_more_info_choice_4_prob = num_high_info_choice_4/total_high_or_low_info_choice_4;
    ff.h5_more_info_choice_5_prob = num_high_info_choice_5/total_high_or_low_info_choice_5;



     
    % get generative mean difference (left - right) for left and right
    % choices, respectively, to determine if value sensitive
    gen_mean_diff_for_right_choices = []; % should be negative
    gen_mean_diff_for_left_choices = []; % should be positive
    for game_num = 1:length(data)
        game = data(game_num,:);
        choices = game.key;
        for (choice_num = 5:length(choices))
            choice = choices(choice_num);
            if choice == 1
                gen_mean_diff_for_left_choices = [gen_mean_diff_for_left_choices game.mean(1) - game.mean(2)];
            else
                gen_mean_diff_for_right_choices = [gen_mean_diff_for_right_choices game.mean(1) - game.mean(2)];
            end
        end
    end

    % t test
    % significance means that person was value-sensitive in the appropriate
    % direction i.e., left-right for left choices was greater than
    % left-right for right choices
    [h, p, ci, stats] = ttest2(gen_mean_diff_for_left_choices, gen_mean_diff_for_right_choices,'Tail', 'right');
    ff.p_value_of_t_test_for_value_sensitivity = p;



    
%     ff.h5_right_30_less = pright(h6_22, -30);
%     ff.h5_right_20_less = pright(h6_22, -20);
%     ff.h5_right_12_less = pright(h6_22, -12);
%     ff.h5_right_08_less = pright(h6_22, -8);
%     ff.h5_right_04_less = pright(h6_22, -4);
%     ff.h5_right_30_more = pright(h6_22, 30);
%     ff.h5_right_20_more = pright(h6_22, 20);
%     ff.h5_right_12_more = pright(h6_22, 12);
%     ff.h5_right_08_more = pright(h6_22, 8);
%     ff.h5_right_04_more = pright(h6_22, 4);
    
%     ff.h1_right_30_less = pright(h1_22, -30);
%     ff.h1_right_20_less = pright(h1_22, -20);
%     ff.h1_right_12_less = pright(h1_22, -12);
%     ff.h1_right_08_less = pright(h1_22, -8);
%     ff.h1_right_04_less = pright(h1_22, -4);
%     ff.h1_right_30_more = pright(h1_22, 30);
%     ff.h1_right_20_more = pright(h1_22, 20);
%     ff.h1_right_12_more = pright(h1_22, 12);
%     ff.h1_right_08_more = pright(h1_22, 8);
%     ff.h1_right_04_more = pright(h1_22, 4);
    % end Figure2A ------------------------------
    
    
    % ---------------------------------------------------------------
    
    ff.mean_RT       = mean([data.RT]);
    ff.sub_accuracy  = mean([data.accuracy]);
    
    ff.choice5_acc_gen_mean      = mean([data.choice5_generative_correct]);
    ff.choice5_acc_obs_mean      = mean([data.choice5_observed_correct]);
    ff.choice5_acc_true_mean     = mean([data.choice5_true_correct]);
    ff.choice5_acc_gen_mean_h5   = mean([h5.choice5_generative_correct]);
    ff.choice5_acc_obs_mean_h5   = mean([h5.choice5_observed_correct]);
    ff.choice5_acc_true_mean_h5  = mean([h5.choice5_true_correct]);
    ff.choice5_acc_gen_mean_h1   = mean([h1.choice5_generative_correct]);
    ff.choice5_acc_obs_mean_h1   = mean([h1.choice5_observed_correct]);
    ff.choice5_acc_true_mean_h1  = mean([h1.choice5_true_correct]);
    
    ff.last_acc_gen_mean         = mean([data.last_generative_correct]);
    ff.last_acc_obs_mean         = mean([data.last_observed_correct]);
    ff.last_acc_true_mean        = mean([data.last_true_correct]);
    ff.last_acc_gen_mean_h5      = mean([h5.last_generative_correct]);
    ff.last_acc_obs_mean_h5      = mean([h5.last_observed_correct]);
    ff.last_acc_true_mean_h5     = mean([h5.last_true_correct]);
    ff.last_acc_gen_mean_h1      = mean([h1.last_generative_correct]);
    ff.last_acc_obs_mean_h1      = mean([h1.last_observed_correct]);
    ff.last_acc_true_mean_h1     = mean([h1.last_true_correct]);

    
    ff.mean_RT_h5                = mean([h5.RT]); 
    ff.mean_RT_h1                = mean([h1.RT]); 
    
    ff.mean_RT_choice5           = mean([data.RT_choice5]);
    ff.mean_RT_choiceLast        = mean([data.RT_choiceLast]);
    
    ff.mean_RT_choice5_h5        = mean([h5.RT_choice5]);
    ff.mean_RT_choiceLast_h5     = mean([h5.RT_choiceLast]);
    ff.mean_RT_choice5_h1        = mean([h1.RT_choice5]);
    ff.mean_RT_choiceLast_h1     = mean([h1.RT_choiceLast]);
    
    ff.true_correct_frac         = mean([data.true_correct_frac]);
    ff.true_correct_frac_h1      = mean([h1.true_correct_frac]);
    ff.true_correct_frac_h5      = mean([h5.true_correct_frac]);

    ff.num_games                 = size(data,1);
    
%     if sum(ismember(input_dir, '1'))==1
%         cb = '1';
%     elseif sum(ismember(input_dir, '2'))==0
%         cb = '2';
%     else
%         cb='';
%     end
%     ff.last_acc_true_mean_h122   = mean([h1_22.last_true_correct]);
    
%    ff = struct2table(ff, AsArray=true);
%  writetable(ff, [
%        results_dir '/' subject '_ses' num2str(ses) '_fit.csv'
%      ])
end

function p = pminfo(hor, amt)
    relev = hor([hor.info_diff] == amt);
    
    if numel(relev) > 0
        minfo = [relev.more_info]';
    %     disp([relev.more_info]);
        keys = vertcat(relev.key);

        p = sum(keys(:, 5) == minfo) / numel(relev);
    else
        p = NaN;
    end
end



% I think this function is wrong because info_diff is the mean difference
% for high - low info options. I want the mean diff for left - right
function p = pright(hor, amt)
    relev = hor([hor.info_diff] == amt);
    if numel(relev) > 0    
        keys = vertcat(relev.key);
        p = sum(keys(:, 5) == 2) / numel(relev);
    else
        p = NaN;
    end
end

% 
% fields = fieldnames(data(1));
% for i = 1:40
%     for field_indx = 1:length(fields)
%         field = fields{field_indx};
%         for k = 1:numel(data(i).(field))
%             old_value = data(i).(field);
%             new_value = new_mf_data(i).(field);
%             if old_value(k) ~= new_value(k)
%                 fprintf("bad");
%             end
%         end
%     end
% end