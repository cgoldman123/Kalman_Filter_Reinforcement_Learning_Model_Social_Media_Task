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

%% Perform model fit
% Reads in the above 'outpath_beh' file to fit
[fits, model_output] = fit_extended_model(outpath_beh, results_dir, MDP);
for i = 1:numel(model_output)
    subject = subj_mapping{i, 1};  
    model_output(i).results.subject = subject{:};
    model_output(i).results.room_type = room_type;
    model_output(i).results.cb = subj_mapping{i, 3};  
end
save(sprintf([results_dir 'model_output_%s_%s.mat'], room_type, timestamp),'model_output');
fits_table.id = string(subj_mapping{:, 1});
fits_table.model = func2str(MDP.model);
fits_table.has_practice_effects = (ismember(fits_table.id, flag));
fits_table.room_type = room_type;
vars = fieldnames(fits);
for i = 1:length(vars)
    if any(strcmp(vars{i}, MDP.field))
        fits_table.(['prior_' vars{i}]) = MDP.params.(vars{i});
        fits_table.(['posterior_' vars{i}]) = fits.(vars{i});
    elseif contains(vars{i}, 'simfit') || strcmp(vars{i}, 'model_acc') || strcmp(vars{i}, 'average_action_prob') ||  strcmp(vars{i}, 'F')
        fits_table.(vars{i}) = fits.(vars{i});
    else
        fits_table.(['fixed_' vars{i}]) = fits.(vars{i});
    end
end

outpath_fits = sprintf([results_dir '%s_fits_%s_%s.csv'], fits_table.id, room_type, timestamp);
writetable(struct2table(fits_table), outpath_fits);
varargout{1} = fits_table;

end
