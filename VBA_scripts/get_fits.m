function varargout = get_fits(root,study,room_type, results_dir, MDP, id)
timestamp = datestr(datetime('now'), 'mm_dd_yy_THH-MM-SS');

% Import matjags library and model-specific fitting function
addpath([root 'rsmith/all-studies/core/matjags']);
addpath([root 'rsmith/all-studies/models/extended-horizon']);

%% Clean up files and concatenate for fitting
files = {};
subs = {id};

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


[big_table, subj_mapping, flag] = Social_merge(subs, files, room_type, study);


outpath_beh = sprintf([results_dir '%s_beh_%s_%s.csv'], id, room_type, timestamp);
writetable(big_table, outpath_beh);

if MDP.fit_model
    try
    
        %% Perform model fit
        % Reads in the above 'outpath_beh' file to fit
        [fits, model_output] = fit_extended_model(outpath_beh, results_dir, MDP);
        for i = 1:numel(model_output)
            subject = subj_mapping{i, 1};  
            model_output(i).results.subject = subject{:};
            model_output(i).results.room_type = room_type;
            model_output(i).results.cb = subj_mapping{i, 3}; 
            model_output.id = subject{:}; model_output.room_type = room_type; model_output.cb = subj_mapping{i, 3};
        end
        id = subj_mapping{1, 1};
        if MDP.get_rts_and_dont_fit_model
            varargout{1} = model_output.results;
            return;
        end
        
        save(sprintf([results_dir 'model_output_%s_%s_%s.mat'], id{:}, room_type, timestamp),'model_output');
        fits_table.id = id;
        fits_table.model = func2str(MDP.model);
        fits_table.has_practice_effects = (ismember(fits_table.id, flag));
        fits_table.room_type = room_type;
        fits_table.fitting_procedure = 'VBA';
        % Add mapping fields if they exist in MDP.settings
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
            if any(strcmp(vars{i}, MDP.field))
                fits_table.(['prior_' vars{i}]) = MDP.params.(vars{i});
                fits_table.(['posterior_' vars{i}]) = fits.(vars{i});
            elseif contains(vars{i}, 'simfit') || strcmp(vars{i}, 'model_acc') || strcmp(vars{i}, 'average_action_prob') ||  strcmp(vars{i}, 'F') || ...
                    strcmp(vars{i},'average_action_prob_H5_1') || strcmp(vars{i},'average_action_prob_H5_2') || strcmp(vars{i},'average_action_prob_H5_3') || ...
                    strcmp(vars{i},'average_action_prob_H5_4') || strcmp(vars{i},'average_action_prob_H5_5') || strcmp(vars{i},'average_action_prob_H1_1') 
                fits_table.(vars{i}) = fits.(vars{i});
            else
                fits_table.(['fixed_' vars{i}]) = fits.(vars{i});
            end
        end
    
    catch ME
        fprintf("Model didn't fit!\n");
        fprintf(2, "ERROR: %s\n", ME.message); % Red text for visibility
        fprintf("Occurred in function: %s\n", ME.stack(1).name);
        fprintf("File: %s\n", ME.stack(1).file);
        fprintf("Line: %d\n", ME.stack(1).line);
    end
end

if MDP.do_model_free
    try
        good_behavioral_file = subj_mapping{1,4};
        model_free = social_model_free(root,good_behavioral_file,room_type,study,struct());
    catch ME
        fprintf("Model free didn't work!");
        fprintf("ERROR: %s\n", ME.message); 
        fprintf("Occurred in function: %s\n", ME.stack(1).name);
        fprintf("File: %s\n", ME.stack(1).file);
        fprintf("Line: %d\n", ME.stack(1).line);    end
end

if MDP.do_simulated_model_free
    try
        good_behavioral_file = subj_mapping{1,4};
        simulated_model_free = social_model_free(root,good_behavioral_file,room_type,study,model_output.simfit_DCM.datastruct);
    catch ME
        fprintf("Simulate model free didn't work!");
        fprintf(2, "ERROR: %s\n", ME.message); % Red text for visibility
        fprintf("Occurred in function: %s\n", ME.stack(1).name);
        fprintf("File: %s\n", ME.stack(1).file);
        fprintf("Line: %d\n", ME.stack(1).line);    end
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
