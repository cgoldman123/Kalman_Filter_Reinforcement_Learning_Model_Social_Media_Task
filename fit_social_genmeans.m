function ff = fit_social_genmeans(file,subject, ses, room_type, input_dir, study)
    %The only difference between the two versions of the schedules is the
    %order of blocks. Since we fit Dislike and Like rooms separately and
    %the same block within each room type always goes first (within a
    %session), we can use the same schedule.
    
     % determine if cb=1 or cb=2
    if strcmp(study,'local')
        if contains(file, '_R1-')
            schedule = readtable('../schedules/sm_distributed_schedule1.csv');
            cb = 1;
        else
            schedule = readtable('../schedules/sm_distributed_schedule1_CB.csv');
            cb = 2;
        end
    elseif strcmp(study,'prolific')
        if contains(file, '_CB_')
            schedule = readtable('../schedules/sm_distributed_schedule1_CB.csv');
            cb = 2;
        else
            schedule = readtable('../schedules/sm_distributed_schedule1.csv');
            cb = 1;
        end
     end
    
    
    
    orgfunc = str2func(['Social_' study '_organize']);
    subj_data = orgfunc(file{:}, schedule, room_type);
    
    % for debugging: disp(subj_data);disp(file{:});disp(ses); disp(room_type);
    data = parse_table(subj_data, file{:}, ses, 80, room_type);
        
    ff = fit_horizon(data, ses, room_type);
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
    %ff.h6_freec6_acc = sum(h6_meancor(:, 10)) / numel(h6);
    
    ff.h1_freec1_acc = sum(h1_meancor(:, 5)) / numel(h1);
    % end Figure1C ------------------------------
    
    % for Figure1D ------------------------------
    % ???
    % end Figure1D ------------------------------
    
    % for Figure2A ------------------------------
    ff.h5_more_info_24_less = pminfo(h5_13, -24);
    ff.h5_more_info_12_less = pminfo(h5_13, -12);
    ff.h5_more_info_08_less = pminfo(h5_13, -8);
    ff.h5_more_info_04_less = pminfo(h5_13, -4);
    ff.h5_more_info_02_less = pminfo(h5_13, -2);
    ff.h5_more_info_24_more = pminfo(h5_13, 24);
    ff.h5_more_info_12_more = pminfo(h5_13, 12);
    ff.h5_more_info_08_more = pminfo(h5_13, 8);
    ff.h5_more_info_04_more = pminfo(h5_13, 4);
    ff.h5_more_info_02_more = pminfo(h5_13, 2);
    
    ff.h1_more_info_24_less = pminfo(h1_13, -24);
    ff.h1_more_info_12_less = pminfo(h1_13, -12);
    ff.h1_more_info_08_less = pminfo(h1_13, -8);
    ff.h1_more_info_04_less = pminfo(h1_13, -4);
    ff.h1_more_info_02_less = pminfo(h1_13, -2);
    ff.h1_more_info_24_more = pminfo(h1_13, 24);
    ff.h1_more_info_12_more = pminfo(h1_13, 12);
    ff.h1_more_info_08_more = pminfo(h1_13, 8);
    ff.h1_more_info_04_more = pminfo(h1_13, 4);
    ff.h1_more_info_02_more = pminfo(h1_13, 2);
    
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
    
    ff = struct2table(ff, AsArray=true);
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

function p = pright(hor, amt)
    relev = hor([hor.info_diff] == amt);
    if numel(relev) > 0    
        keys = vertcat(relev.key);
        p = sum(keys(:, 5) == 2) / numel(relev);
    else
        p = NaN;
    end
end