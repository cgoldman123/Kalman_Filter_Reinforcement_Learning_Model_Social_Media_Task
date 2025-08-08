function [all_data, subject_data_info] = get_raw_data_EIT(root, ids, groupdata)        
    
    folder = [root '/rsmith/lab-members/cgoldman/EIT_horizon/data/'];
    
    [all_data, raw_path] = EIT_compile_data(folder, ids, groupdata);  

    

    all_data.subjectID = repmat(1, size(all_data, 1), 1); % just set it as 1, doesn't matter much
    

    subject_data_info.id = ids;
    subject_data_info.behavioral_file_path = raw_path;
    
end