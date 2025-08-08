function [raw_data,raw_file_path] = get_raw_data(root,study,room_type,id)
    % Clean up files and concatenate for fitting
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
        sink = [root '/NPC/DataSink/COBRE-NEUT/data-original/behavioral_session/'];
        all_files = dir(fullfile(sink, '**', '*'));  % recursive search
        all_files = all_files(~[all_files.isdir]);   % only files, not folders
        % Check for filename matches
        file_names = {all_files.name};
        matches = contains(file_names, id) & contains(file_names, 'HZ') & ~contains(file_names, "PR");
        % Show full paths of matched files
        matched_files = fullfile({all_files(matches).folder}, {all_files(matches).name});
        
        if length(files) > 1
            fprintf("MULTIPLE FILES FOUND FOR THIS LOCAL PARTICIPANT.");
            file = matched_files;
        else
            file = matched_files;
        end
    
    end
    
    
    [raw_data, subj_mapping, ~] = Social_merge(subs, file, room_type, study);
    raw_file_path = subj_mapping{1,4};
    
end