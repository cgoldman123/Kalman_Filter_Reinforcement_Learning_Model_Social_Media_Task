function trial_by_trial_output = get_trial_by_trial_output(model_output, subject_data_info,processed_data)

    % Use this function to extract the trial by trial output from the model
    % (e.g., prediction errors) and behavior e.g., reaction times, choices

    % Determine number of games and trials
    num_games = processed_data.num_games;
    num_trials = processed_data.num_forced_choices + processed_data.num_free_choices_big_hor;
    
    % Preallocate repeating variables
    id_col     = repmat({subject_data_info.id}, num_games * num_trials, 1);
    study_col     = repmat({subject_data_info.study}, num_games * num_trials, 1);
    condition_col  = repmat({subject_data_info.condition}, num_games * num_trials, 1);
    experiment_col  = repmat({subject_data_info.experiment}, num_games * num_trials, 1);
    run_col  = repmat({subject_data_info.run}, num_games * num_trials, 1);
    room_col   = repmat({subject_data_info.room_type}, num_games * num_trials, 1);
    cb_col     = repmat({subject_data_info.cb}, num_games * num_trials, 1);
    
    % Create game and trial indices
    game_col   = repelem((1:num_games)', num_trials);
    trial_col  = repmat((1:num_trials)', num_games, 1);

    % Create horizon col
    horizon_vals = processed_data.horizon_type;
    horizon_vals(horizon_vals == 2) = processed_data.num_free_choices_big_hor;
    horizon_col = repelem(horizon_vals, num_trials);

    
    % Helper function to extract 40x9 subset from 40x10 or keep as-is
    extract_trials = @(mat) mat(:, 1:num_trials);
    
    % Flatten processed_data variables
    actions_col = reshape(extract_trials(processed_data.actions)', [], 1);
    rewards_col = reshape(extract_trials(processed_data.rewards)', [], 1);
    RTs_col     = reshape(extract_trials(processed_data.RTs)', [], 1);
    
    % Flatten model_output variables
    exp_vals_col                  = reshape(extract_trials(model_output.exp_vals)', [], 1);
    pred_errors_col              = reshape(extract_trials(model_output.pred_errors)', [], 1);
    pred_errors_alpha_col        = reshape(extract_trials(model_output.pred_errors_alpha)', [], 1);
    relative_uncertainty_col     = reshape(extract_trials(model_output.relative_uncertainty_of_choice)', [], 1);
    total_uncertainty_col        = reshape(extract_trials(model_output.total_uncertainty)', [], 1);
    delta_uncertainty_col        = reshape(extract_trials(model_output.change_in_uncertainty_after_choice)', [], 1);
    
    % Create final table
    trial_by_trial_output = table(...
        id_col, study_col, condition_col, experiment_col, run_col, room_col, cb_col, game_col, trial_col, horizon_col,...
        actions_col, rewards_col, RTs_col, ...
        exp_vals_col, pred_errors_col, pred_errors_alpha_col, ...
        relative_uncertainty_col, total_uncertainty_col, delta_uncertainty_col, ...
        'VariableNames', {...
            'id', 'study','condition','experiment','run', 'room_type', 'cb', 'game_num', 'trial_num', 'horizon', ...
            'action', 'reward', 'RT', ...
            'exp_val', 'pred_error', 'pred_error_alpha', ...
            'relative_uncertainty', 'total_uncertainty', 'delta_uncertainty' ...
        });


end