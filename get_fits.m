function varargout = get_fits(root, study,room_type, results_dir, MDP, id)
timestamp = datestr(datetime('now'), 'mm_dd_yy_THH-MM-SS');
[raw_data,raw_file_path] = get_raw_data(root,study,room_type,id);


if MDP.fit_model
    [fits, model_output] = fit_extended_model_SPM(outpath_beh, results_dir, MDP);

    for i = 1:numel(model_output)
        subject = id;  
        model_output(i).results.subject = subject{:};
        model_output(i).results.room_type = room_type;
        model_output(i).results.cb = subj_mapping{i, 3}; 
        model_output(i).results.good_behavioral_file = good_behavioral_file;
        model_output.id = subject{:}; 
        model_output.room_type = room_type; 
        model_output.cb = subj_mapping{i, 3};
    end
    id = subj_mapping{1, 1};
    if MDP.get_processed_behavior_and_dont_fit_model
        varargout{1} = model_output.results;
        varargout{2} = model_output.datastruct;
        return;
    end
    
    save(sprintf([results_dir 'model_output_%s_%s_%s.mat'], id{:}, room_type, timestamp),'model_output');
    fits_table.id = id;
    fits_table.has_practice_effects = (ismember(fits_table.id, flag));
    fits_table.room_type = room_type;
    fits_table.model = func2str(MDP.model);
    if isfield(MDP.settings, 'drift_mapping')
        if isempty(MDP.settings.drift_mapping)
            fits_table.drift_mapping = ' ';
        else
            fits_table.drift_mapping = strjoin(MDP.settings.drift_mapping);
        end
    end
    
    if isfield(MDP.settings, 'bias_mapping')
        if isempty(MDP.settings.bias_mapping)
            fits_table.bias_mapping = ' ';
        else
            fits_table.bias_mapping = strjoin(MDP.settings.bias_mapping);
        end
    end
    
    if isfield(MDP.settings, 'thresh_mapping')
        if isempty(MDP.settings.thresh_mapping)
            fits_table.thresh_mapping = ' ';
        else
            fits_table.thresh_mapping = strjoin(MDP.settings.thresh_mapping);
        end
    end

    vars = fieldnames(fits);
    for i = 1:length(vars)
        % If the variable is a free parameter, assign prior and
        % posterior vals
        if any(strcmp(vars{i}, MDP.field))
            fits_table.(['prior_' vars{i}]) = MDP.params.(vars{i});
            fits_table.(['posterior_' vars{i}]) = fits.(vars{i});
        % If the variable is not a free param but relates to the
        % model, assign it to fits_table
        elseif contains(vars{i}, 'simfit') || strcmp(vars{i}, 'num_invalid_rts') || strcmp(vars{i}, 'model_acc') || contains(vars{i}, 'average_action_prob') ||  strcmp(vars{i}, 'F')
            fits_table.(vars{i}) = fits.(vars{i});
        % Otherwise, the variable must be one of the parameters 
        % that was fixed (not fit) so include the fixed prefix 
        else
            fits_table.(['fixed_' vars{i}]) = fits.(vars{i});
        end
    end
end

if MDP.do_model_free
    model_free = social_model_free(root,good_behavioral_file,room_type,study,struct());
end

if MDP.do_simulated_model_free
    simulated_model_free = social_model_free(root,good_behavioral_file,room_type,study,model_output.simfit_DCM.datastruct);
end

% if fits and/or model-free analyses are present, add them to the output
% struct
if exist('fits_table','var')
    output = fits_table;
    if exist('model_free','var')
        % Get field names of both structs
        fits_fields = fieldnames(fits_table);
        model_free_fields = fieldnames(model_free);
        % Loop through each field in model_free
        for i = 1:length(model_free_fields)
            field_name = model_free_fields{i};
            % If the field is not in fits_table, add it to output
            if ~ismember(field_name, fits_fields)
                output.(field_name) = model_free.(field_name);
            end
        end
    end
    if exist('simulated_model_free','var')
        % Get field names of both structs
        fits_fields = fieldnames(fits_table);
        simulated_model_free_fields = fieldnames(simulated_model_free);
        % Loop through each field in model_free
        for i = 1:length(simulated_model_free_fields)
            field_name = simulated_model_free_fields{i};
            % If the field is not in fits_table, add it to output
            if ~ismember(field_name, fits_fields)
                output.(['simulated_' field_name]) = simulated_model_free.(field_name);
            end
        end
    end
else
    if exist('model_free','var')
        output = struct();
        output.id = {id};
        model_free_fields = fieldnames(model_free);
        % Loop through each field in model_free
        for i = 1:length(model_free_fields)
            field_name = model_free_fields{i};        
            output.(field_name) = model_free.(field_name);
        end
    end
end


outpath_fits = sprintf([results_dir '%s_fits_%s_%s.csv'], output.id{:}, room_type, timestamp);
writetable(struct2table(output,'AsArray',true), outpath_fits);
varargout{1} = output;

end
