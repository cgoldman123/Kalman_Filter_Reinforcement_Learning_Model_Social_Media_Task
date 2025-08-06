function varargout = get_fits(root, fitting_procedure, study,room_type, results_dir, MDP, id)
timestamp = datestr(datetime('now'), 'mm_dd_yy_THH-MM-SS');

%% Clean up files and concatenate for fitting
files = {};
subs = {id};

if strcmp(study, 'prolific')
    file_path = fullfile(root, 'NPC/DataSink/StimTool_Online/WB_Social_Media/');
    SM_directory = dir(file_path);
    index_array = find(arrayfun(@(n) contains(SM_directory(n).name, strcat('social_media_',id)),1:numel(SM_directory)));
    for k = 1:length(index_array)
        file_index = index_array(k);
        files{k} = [file_path SM_directory(file_index).name];
    end
    
    file_path_cb = fullfile(root, 'NPC/DataSink/StimTool_Online/WB_Social_Media_CB/');
    SM_directory = dir(file_path_cb);
    index_array_cb = find(arrayfun(@(n) contains(SM_directory(n).name, strcat('social_media_',id)),1:numel(SM_directory)));
    for k = 1:length(index_array_cb)
        file_index_cb = index_array_cb(k);
        files{length(files)+k} = [file_path_cb SM_directory(file_index_cb).name];
    end
    
    if exist('file_index', 'var') & exist('file_index_cb', 'var')
        error("Participant has files for CB1 and CB2");
    end
elseif strcmp(study, 'local')
    sink = [root 'rsmith/wellbeing/data/raw/'];
    all_subfolders = dir(sink);
    % Filter to find the subfolder for this ID
    is_match = strcmp({all_subfolders.name}, ['sub-' id]) & [all_subfolders.isdir];
    match_folder = all_subfolders(is_match);
    for i = 1:length(match_folder)
        file_list = dir([sink match_folder(i).name]);
        file_list = file_list(~[file_list.isdir]);  
        for file_idx = 1:length(file_list)
            if ~isempty(regexp(file_list(file_idx).name, 'SM_R[0-9]-_BEH', 'once'))
                files = [files [sink match_folder(i).name '/' file_list(file_idx).name]];
            end
        end
    end

    if length(files) > 1
        fprintf("MULTIPLE FILES FOUND FOR THIS LOCAL PARTICIPANT.");
    end

end


[big_table, subj_mapping, flag] = Social_merge(subs, files, room_type, study);
good_behavioral_file = subj_mapping{1,4};

outpath_beh = sprintf([results_dir '%s_beh_%s_%s.csv'], id, room_type, timestamp);
writetable(big_table, outpath_beh);

if MDP.fit_model
    try
    
        %% Perform model fit
        % Reads in the above 'outpath_beh' file to fit
        if strcmp(fitting_procedure, "VBA")
            [fits, model_output] = fit_extended_model_no_latent_state_learning(outpath_beh, results_dir, MDP);
        elseif strcmp(fitting_procedure, "SPM")
            [fits, model_output] = fit_extended_model_SPM(outpath_beh, results_dir, MDP);
        elseif strcmp(fitting_procedure, "PYDDM")
            % call python scripts from inside matlab. Note that this
            % environment must have the pyddm package installed
            if ispc
                % Use the python path from the conda virtual environment
                python_path = 'C:\Users\CGoldman\AppData\Local\anaconda3\envs\pyddm\python.exe';
            else
                % Use the python path from the conda virtual environment.
                % In the .ssub file, I load this conda environment.
                python_path = '/mnt/dell_storage/homefolders/librad.laureateinstitute.org/cgoldman/miniconda3/envs/pyddm/bin/python';
            end
            script_path = [root 'rsmith/lab-members/cgoldman/Wellbeing/social_media/scripts/PyDDM_scripts/fit_pyddm_social_media.py'];
            % Build the command safely (quotes to handle spaces)
            cmd = sprintf('"%s" "%s" "%s" "%s" "%s" "%s" "%s" "%s"', ...
                python_path, script_path, outpath_beh, results_dir, id, room_type, timestamp, MDP.settings);
            
            % Run and display output
            [status, cmdout] = system(cmd);
            disp(cmdout)

            % Load the resulting matlab object.
            pyddm_results = load([results_dir id '_' room_type '_' timestamp '_model_results_pyddm.mat']);
            fits = pyddm_results.fit_result;
            model_output = pyddm_results.model_output;

        end
        for i = 1:numel(model_output)
            subject = subj_mapping{i, 1};  
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
        fits_table.fitting_procedure = fitting_procedure;
        % Add the parameter estimates and mapping fields for non-PYDDM fits
        if ~strcmp(fitting_procedure, "PYDDM")
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
        else
            % PYDDM fits

            % Add any settings passed in through the field variable
            fits_table.settings = MDP.settings;
            % Note that the prefix sft means the result was from fitting
            % simulated data.
            vars = fieldnames(fits);
            for i = 1:length(vars)
                fits_table.(vars{i}) = fits.(vars{i});
            end


        end
        
       
    
    catch ME
        fprintf("Model didn't fit!\n");
        fprintf("ERROR: %s\n", ME.message); % Red text for visibility
        fprintf("Occurred in function: %s\n", ME.stack(1).name);
        fprintf("File: %s\n", ME.stack(1).file);
        fprintf("Line: %d\n", ME.stack(1).line);
    end
end

if MDP.do_model_free
    try
        model_free = social_model_free(root,good_behavioral_file,room_type,study,struct());
    catch ME
        fprintf("Model free didn't work!");
        fprintf("ERROR: %s\n", ME.message); 
        fprintf("Occurred in function: %s\n", ME.stack(1).name);
        fprintf("File: %s\n", ME.stack(1).file);
        fprintf("Line: %d\n", ME.stack(1).line);    
    end
end

if MDP.do_simulated_model_free
    try
        if strcmp(fitting_procedure, "VBA")
            simulated_model_free = social_model_free(root,good_behavioral_file,room_type,study,model_output.simfit_out.simfit_datastruct);
        elseif strcmp(fitting_procedure, "SPM")
            simulated_model_free = social_model_free(root,good_behavioral_file,room_type,study,model_output.simfit_DCM.datastruct);
        elseif strcmp(fitting_procedure, "PYDDM")
            simulated_model_free = social_model_free(root,good_behavioral_file,room_type,study,model_output.sft_datastruct);
        end
    catch ME
        fprintf("Simulate model free didn't work!");
        fprintf("ERROR: %s\n", ME.message); % Red text for visibility
        fprintf("Occurred in function: %s\n", ME.stack(1).name);
        fprintf("File: %s\n", ME.stack(1).file);
        fprintf("Line: %d\n", ME.stack(1).line);    
    end
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
