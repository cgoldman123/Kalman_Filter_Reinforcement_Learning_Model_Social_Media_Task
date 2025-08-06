function varargout = get_fits(root, processed_data, subject_data_info, results_dir, MDP)
    id = subject_data_info.id;
    room_type = subject_data_info.room_type;
    study = subject_data_info.study;
    timestamp = datestr(datetime('now'), 'mm_dd_yy_THH-MM-SS');
    
    
    if MDP.fit_model
        [fits, model_output] = fit_extended_model_SPM(processed_data, MDP);
    
        % Assemble .mat object to save
        model_output.id = id; 
        model_output.room_type = room_type;
        model_output.cb = subject_data_info.cb;
        model_output.raw_behavioral_file = subject_data_info.behavioral_file_path;
        save(sprintf([results_dir 'model_output_%s_%s_%s.mat'], id, room_type, timestamp),'model_output');
    
        % Assemble fits table to save as CSV
        fits_table.id = id;
        fits_table.cb = subject_data_info.cb;
        fits_table.has_practice_effects = subject_data_info.has_practice_effects;
        fits_table.room_type = room_type;
        fits_table.model = func2str(MDP.model);
        fits_table.raw_behavioral_file = subject_data_info.behavioral_file_path;
       
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
        model_free = social_model_free(root,subject_data_info.behavioral_file_path,room_type,study,struct());
    end
    
    if MDP.do_simulated_model_free
        simulated_model_free = social_model_free(root,subject_data_info.behavioral_file_path,room_type,study,model_output.simfit_DCM.processed_data);
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
    
    
    outpath_fits = sprintf([results_dir '%s_fits_%s_%s.csv'], id, room_type, timestamp);
    writetable(struct2table(output,'AsArray',true), outpath_fits);
    varargout{1} = output;
    
end
