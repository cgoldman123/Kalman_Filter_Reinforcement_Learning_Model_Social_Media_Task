function     [processed_data,raw_data,subject_data_info] = process_data_across_studies(root,study_info)
    %%%% Specify the data to process for the exercise study
    if strcmp(study_info.study, "wellbeing")
        [raw_data,subject_data_info] = get_raw_data_SM(root,study_info.experiment,study_info.room,study_info.id);
        processed_data = process_behavioral_data_SM(raw_data);

    %%%% Specify the data to process for the exercise study
    elseif strcmp(study_info.study,'exercise')
        [raw_data,subject_data_info] = get_raw_data_exercise(root,study_info.id,study_info.run,study_info.room);
        processed_data = process_behavioral_data_exercise(raw_data);

    %%%% Specify the data to process for the cobre_neut study
    elseif strcmp(study_info.study,'cobre_neut')
        [raw_data,subject_data_info] = Berg_get_raw_data(root,study_info.room,study_info.id);
        processed_data = process_behavioral_data_Berg(raw_data);

    %%%% Specify the data to process for the adm study
    elseif strcmp(study_info.study,'adm')
        group_list = readtable("./data_processing/adm_data_processing/group_list.csv"); % hor_task_counterbalance- 1: loaded first, 2: unloaded first
        [raw_data, subject_data_info] = get_raw_data_ADM(root, study_info.id, group_list, study_info.condition);
        processed_data = process_behavioral_data_ADM(raw_data);

    %%%% Specify the data to process for the eit study
    elseif strcmp(study_info.study,'eit')
        group_list = readtable("./data_processing/EIT_data_processing/EIT_subject_and_notes.csv"); % hor_task_counterbalance- 1: loaded first, 2: unloaded first
        [raw_data, subject_data_info] = get_raw_data_EIT(root, study_info.id, group_list);
        processed_data = process_behavioral_data_EIT(raw_data);
    end

end