function output_struct = get_simulated_model_free(processed_data, MDP,subject_data_info,root,results_dir);
    id = subject_data_info.id;
    room_type = subject_data_info.room_type;
    experiment = subject_data_info.study;
    timestamp = datestr(datetime('now'), 'mm_dd_yy_THH-MM-SS');

    MDP.processed_data = processed_data;

    num_choices_big_hor = processed_data.num_forced_choices + processed_data.num_free_choices_big_hor;
    actions_and_rts.actions = processed_data.actions;
    actions_and_rts.RTs = nan(processed_data.num_games,num_choices_big_hor);

    simmed_model_output = MDP.model(MDP.params, actions_and_rts, processed_data.rewards, MDP, 1);

    processed_data.actions = simmed_model_output.actions;
    processed_data.rewards = simmed_model_output.rewards;
    processed_data.RTs = simmed_model_output.rts;


    % Build output struct and fill with parameter values
    output_struct.id = id;
    param_fields = fieldnames(MDP.params);
    for i = 1:length(param_fields)
        output_struct.(param_fields{i}) = MDP.params.(param_fields{i});
    end
    output_struct.field = strjoin(MDP.field, ','); 

    % Call model free function
    simulated_model_free = social_model_free(processed_data);
    % Continue building output struct with model free stats
    simulated_model_free_fields = fieldnames(simulated_model_free);
    for i = 1:length(simulated_model_free_fields)
        output_struct.(simulated_model_free_fields{i}) = simulated_model_free.(simulated_model_free_fields{i});
    end


    outpath = sprintf([results_dir 'simulated_model_free_%s_%s_%s.csv'], id, room_type, timestamp);
    writetable(struct2table(output_struct,'AsArray', true), outpath);
    
end