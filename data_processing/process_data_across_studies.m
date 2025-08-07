function     [processed_data,raw_data,subject_data_info] = process_data_across_studies(root,study_info)
    if strcmp(study_info.study, "wellbeing")
        [raw_data,subject_data_info] = get_raw_data_SM(root,study_info.experiment,study_info.room,study_info.id);
        processed_data = process_behavioral_data_SM(raw_data);
    end
end