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
    
    
    [raw_data, subj_mapping, ~] = Social_merge(subs, files, room_type, study);
    raw_file_path = subj_mapping{1,4};
    
end