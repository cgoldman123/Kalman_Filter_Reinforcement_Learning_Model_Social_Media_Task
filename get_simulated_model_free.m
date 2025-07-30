function output_struct = get_simulated_model_free(root, fitting_procedure, experiment, room_type, cb, results_dir,MDP,id)
    timestamp = datestr(datetime('now'), 'mm_dd_yy_THH-MM-SS');

    % First call get_fits to get the schedule/forced choices before
    MDP.get_processed_behavior_and_dont_fit_model = 1; % Toggle on to extract the rts and other processed behavioral data but not fit the model
    MDP.fit_model = 1; % Toggle on even though the model won't fit
    [rt_data, mdp] = get_fits(root, fitting_procedure, experiment,room_type, results_dir, MDP, id);
    

    mdp_fieldnames = fieldnames(mdp);
    for (i=1:length(mdp_fieldnames))
        MDP.(mdp_fieldnames{i}) = mdp.(mdp_fieldnames{i});
    end
    actions_and_rts.actions = mdp.actions;
    actions_and_rts.RTs = nan(40,9);
    simmed_model_output = MDP.model(MDP.params, actions_and_rts, MDP.rewards, MDP, 1);

    datastruct.actions = simmed_model_output.actions;
    datastruct.rewards = simmed_model_output.rewards;
    
    if ismember(func2str(MDP.model), {'model_SM_KF_DDM_all_choices', 'model_SM_KF_SIGMA_DDM_all_choices'})
        datastruct.RTs = simmed_model_output.rts;
    else
        datastruct.RTs = nan(40,9);
    end

    % Load in example file to get schedule
    if strcmp(experiment, 'prolific')
        if cb==1
            file = [root '\NPC\DataSink\StimTool_Online\WB_Social_Media\social_media_667fb298629d2b1d2c7ac461_T1_2024-07-31_16h28.13.853.csv'];
        else
            file = [root '\NPC\DataSink\StimTool_Online\WB_Social_Media_CB\social_media_666878a27888fdd27f529c64_T1_CB_2024-08-06_09h14.52.614.csv'];
        end
    else
        % Local data
        if cb==1
            file = [root '\rsmith\wellbeing\data\raw\sub-AV841\AV841-T1-__SM_R1-_BEH.csv'];
        else
            file = [root '\rsmith\wellbeing\data\raw\sub-AA003\AA003-T1-__SM_R3-_BEH.csv'];
        end
    end



    output_struct.id = id;
    param_fields = fieldnames(MDP.params);
    for i = 1:length(param_fields)
        output_struct.(param_fields{i}) = MDP.params.(param_fields{i});
    end
    if isfield(MDP,'settings')
        output_struct.drift_mapping = strjoin(MDP.settings.drift_mapping,",");
        output_struct.bias_mapping = strjoin(MDP.settings.bias_mapping,",");
        output_struct.thresh_mapping = strjoin(MDP.settings.thresh_mapping,",");
    end
    output_struct.field = strjoin(MDP.field, ','); 
    simulated_model_free = social_model_free(root,file,room_type,experiment,datastruct);
    simulated_model_free_fields = fieldnames(simulated_model_free);
    for i = 1:length(simulated_model_free_fields)
        output_struct.(simulated_model_free_fields{i}) = simulated_model_free.(simulated_model_free_fields{i});
    end


    outpath = sprintf([results_dir 'simulated_model_free_%s_%s_%s.csv'], id, room_type, timestamp);
    writetable(struct2table(output_struct,'AsArray', true), outpath);
    
end