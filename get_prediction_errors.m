function all_pred_errors = get_prediction_errors()
    model_outputs = struct();
    for r = {'Like', 'Dislike'}
        directoryPath = ['L:\rsmith\lab-members\cgoldman\Wellbeing\social_media\output\SM_fits_local_7-26-24\local\kf\' r{1}];
        % Get a list of all .mat files in the directory that contain 'model_output'
        files = dir(fullfile(directoryPath, '*model_output*.mat'));
        model_output = load(fullfile(directoryPath, files(1).name));
        model_outputs.(r{1}) = model_output.model_output;
    end

    NS = length(model_outputs.Like); 
    all_pred_errors = table();

    for s=1:NS
        like_output = model_outputs.Like(s).results;
        dislike_output = model_outputs.Dislike(s).results;
        subject = like_output.subject;
        if ~strcmp(subject, dislike_output.subject)
            exception = MException('MyComponent:customException', 'Subject in like output and dislike output are different at the same index.');
            throw(exception);
        end
        if like_output.cb == 1
            schedule = readtable('../schedules/sm_distributed_schedule_CB1.csv');
        else
            schedule = readtable('../schedules/sm_distributed_schedule_CB2.csv');
        end
        % get like trials that were H5
%         like_h5 = unique(schedule.game_number(~schedule.dislike_room & strcmp(schedule.game_type, 'h6')))+1;
%         dislike_h5 = unique(schedule.game_number(schedule.dislike_room & strcmp(schedule.game_type, 'h6')))+1;
        
%         h1 = unique(schedule.game_number(strcmp(schedule.game_type, 'h1')))+1;
%         h5 = unique(schedule.game_number(strcmp(schedule.game_type, 'h6')))+1;


        pred_errors = NaN(560, 1);
        like_errors = reshape(like_output.pred_errors, [], 1);
        dislike_errors = reshape(dislike_output.pred_errors, [], 1);

        pred_errors_alpha = NaN(560, 1);
        like_errors_alpha = reshape(like_output.pred_errors_alpha, [], 1);
        dislike_errors_alpha = reshape(dislike_output.pred_errors_alpha, [], 1);       
        
        exp_vals = NaN(560, 1);
        like_exp_vals = reshape(like_output.exp_vals, [], 1);
        dislike_exp_vals = reshape(dislike_output.exp_vals, [], 1);    
    
        alpha = NaN(560, 1);
        like_alpha = reshape(like_output.alpha, [], 1);
        dislike_alpha = reshape(dislike_output.alpha, [], 1);  
        
        
        trial_counter = 0;
        up_to_five_counter = 0;
        like_counter = 1;
        dislike_counter = 1;
        for i=1:560
            if (schedule.dislike_room(i) == 1)
                if (strcmp(schedule.game_type{i}, 'h1'))
                    pred_errors(i) = dislike_errors(dislike_counter);
                    pred_errors_alpha(i) = dislike_errors_alpha(dislike_counter);
                    exp_vals(i) = dislike_exp_vals(dislike_counter);
                    alpha(i) = dislike_alpha(dislike_counter);

                    dislike_counter = dislike_counter+1;

                else
                    if up_to_five_counter < 5
                        pred_errors(i) = dislike_errors(dislike_counter);
                        pred_errors_alpha(i) = dislike_errors_alpha(dislike_counter);
                        exp_vals(i) = dislike_exp_vals(dislike_counter);
                        alpha(i) = dislike_alpha(dislike_counter);

                        up_to_five_counter = up_to_five_counter+1;
                        dislike_counter = dislike_counter+1;
                    else
                        if up_to_five_counter < 8
                            pred_errors(i) = nan;
                            pred_errors_alpha(i) = nan;
                            exp_vals(i) = nan;
                            alpha(i) = nan;
                                
                            up_to_five_counter = up_to_five_counter+1;
                        else
                            pred_errors(i) = nan;
                            pred_errors_alpha(i) = nan;
                            exp_vals(i) = nan;
                            alpha(i) = nan;

                            up_to_five_counter = 0;
                        end
                    end
                end
            else
                if (strcmp(schedule.game_type{i}, 'h1'))
                    pred_errors(i) = like_errors(like_counter);
                    pred_errors_alpha(i) = like_errors_alpha(like_counter);
                    exp_vals(i) = like_exp_vals(like_counter);
                    alpha(i) = like_alpha(like_counter);

                    like_counter = like_counter+1;

                else
                    if up_to_five_counter < 5
                        pred_errors(i) = like_errors(like_counter);
                        pred_errors_alpha(i) = like_errors_alpha(like_counter);
                        exp_vals(i) = like_exp_vals(like_counter);
                        alpha(i) = like_alpha(like_counter);

                        up_to_five_counter = up_to_five_counter+1;
                        like_counter = like_counter+1;
                    else
                        if up_to_five_counter < 8
                            pred_errors(i) = nan;
                            pred_errors_alpha(i) = nan;
                            exp_vals(i) = nan;
                            alpha(i) = nan;

                            up_to_five_counter = up_to_five_counter+1;
                        else
                            pred_errors(i) = nan;
                            pred_errors_alpha(i) = nan;
                            exp_vals(i) = nan;
                            alpha(i) = nan;

                            up_to_five_counter = 0;
                        end
                    end
                end
            end
        end
        
        subject_name_repeated =  repmat({subject}, size(pred_errors));
        trials =  num2cell((1:size(pred_errors, 1))');
        pred_error_table = struct('subject', subject_name_repeated, 'trial', trials, 'pred_errors', num2cell(pred_errors), ...
                            'alpha', num2cell(alpha), 'pred_errors_alpha', num2cell(pred_errors_alpha), 'exp_vals', num2cell(exp_vals));
        all_pred_errors = vertcat(all_pred_errors, struct2table(pred_error_table));
    end

    timestamp = datestr(datetime('now'), 'mm_dd_yy_THH-MM-SS');
    outpath = sprintf(['L:/rsmith/wellbeing/tasks/SocialMedia/output/local/kf/pred_errors_%s.csv'], timestamp);
    writetable(all_pred_errors, outpath);

end