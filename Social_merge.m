function [all_data, subj_mapping, flag_ids] = Social_merge(ids, files, room_type, study)        
    
    % Note that the ids argument will be as long as the
    % total number of files for all subjects (in the files argument). So there may be 
    % ID repetitions if one ID has multiple behavioral files.  
    
    % This function returns two outputs, all_data and subj_mapping, that
    % will only contain valid subject data. 
   
    % Data is considered valid if it is
    % complete and there are no practice effects (i.e., the subject did not
    % previously start the game). Files are in date order.
    
    if ispc
        root = 'L:/';
    else
        root = '/media/labs/';
    end
    
    
    
    
    all_data = cell(1, numel(ids)); 
    flag_ids = {};
    good_index = [];
    
    subj_mapping = cell(numel(ids), 2); 
    
    for i = 1:numel(ids)
        id   = ids{i};
         % only process this ID if haven't previously processed this ID
         % already
        previously_processed_ids = string(ids(1:i-1));
        if ismember(string(id), previously_processed_ids)
            continue;
        end
        file = files(contains(files, id));    
        success=0;
        has_started_a_game = 0;
        for j = 1:numel(file)
            if ~success
                if strcmp(study,'local')
                    % determine if cb=1 or cb=2; CB1 contains like first,
                    % CB2 contains dislike first
                    filename = file{j};
                    if contains(filename, '_R1-')
                        schedule = readtable([root 'rsmith/wellbeing/tasks/SocialMedia/schedules/sm_distributed_schedule_CB1.csv']);
                        cb = 1;
                    else % CB2 will contain _R3-
                        schedule = readtable([root 'rsmith/wellbeing/tasks/SocialMedia/schedules/sm_distributed_schedule_CB2.csv']);
                        cb = 2;
                    end
                    [all_data{i},started_this_game] = Social_local_parse(filename, schedule, room_type, study);  
                elseif strcmp(study,'prolific')
                    % determine if cb=1 or cb=2; CB1 contains like first,
                    % CB2 contains dislike first
                    filename = file{j};
                    if contains(filename, '_CB_')
                        cb = 2;
                        schedule = readtable([root 'rsmith/wellbeing/tasks/SocialMedia/schedules/sm_distributed_schedule_CB2.csv']);
                    else
                        schedule = readtable([root 'rsmith/wellbeing/tasks/SocialMedia/schedules/sm_distributed_schedule_CB1.csv']);
                        cb = 1;
                    end
                    % note how we still use social_local_parse for prolific
                    [all_data{i},started_this_game] = Social_local_parse(filename, schedule, room_type, study);  
                end
                has_started_a_game = has_started_a_game+started_this_game;
            else
                % continue because we've already found a complete file for
                % this subject (i.e. success==1)
                continue
            end
            
            % this is a good file if it is complete and there are no
            % practice effects
            if((size(all_data{i}, 1) == 40) && (sum(all_data{i}.gameLength) == 280) && (has_started_a_game <= 1))
                good_index = [good_index i];
                success=1;
            end
            
            all_data{i}.subjectID = repmat(i, size(all_data{i}, 1), 1);
            
            subj_mapping{i, 1} = {id};
            subj_mapping{i, 2} = i;
            subj_mapping{i, 3} = cb;
        end
    end
    
    % only take the rows of all_data that are good
    all_data = all_data(good_index);
    subj_mapping = subj_mapping(good_index, :);
    
    all_data = vertcat(all_data{:});    
end