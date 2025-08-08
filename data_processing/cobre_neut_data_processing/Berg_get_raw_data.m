function [raw_data,subject_data_info] = Berg_get_raw_data(root,room_type,id)
    % Clean up files and concatenate for fitting
    files = {};
    subs = {id};
    
    study = 'local';
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
    
    
 
    
    [raw_data, subject_data_info] = Berg_merge(root,subs, file, room_type, study);
    subject_data_info.room_type = room_type;
    
end