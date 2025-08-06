% Function to get reaction times in Social Media Task


%%%%%
% Note that in order to run this you must first ensure that the following
% settings are put in the Social_wrapper.m file:

% MDP.get_rts_and_dont_fit_model = 1; 
% fitting_procedure = "SPM"; 
% experiment = 'prolific';

%%%%%


% Read the CSV file containing IDs
ids = readtable('L:\rsmith\lab-members\cgoldman\Wellbeing\social_media\social_media_prolific_IDs.csv');

% Initialize empty table to store results
all_results = table();

% Loop through each ID

for room = {'Like', 'Dislike'}
    for i = 1:height(ids)
        try
            % Get current ID
            current_id = char(ids.id(i));

            % skip if id is less than 24 characters
            if length(current_id) < 24
                continue;
            end
            
            % Call Social_wrapper with the ID and capture output
            fits_table = Social_wrapper(current_id, room{1});

            % extract RTs from all trials;
            % Script will error if 120 RTs not extracted
            RT_clean = fits_table.RT(~isnan(fits_table.RT));
            free_choices = fits_table.choices(:,5:end); % Get free choices
            free_choices_clean = free_choices(~isnan(free_choices)); % remove NaN values
            id = repmat({fits_table.subject}, 120, 1);   % Repeat subject 120 times as a cell array
            room_type = repmat({fits_table.room_type}, 120, 1); % Repeat room_type 120 times as a cell array
            cb = repmat(fits_table.cb, 120, 1);           % Repeat cb 120 times as a numeric array
            rts_table = table(id, room_type, cb, RT_clean, free_choices_clean);
            % Append results to main table if successful
            if ~isempty(fits_table)
                all_results = [all_results; rts_table];
            end


            
        catch ME
            % If there's an error, print it and continue
            fprintf('Error processing ID %s: %s\n', current_id, ME.message);
            continue;
        end
    end
end

% Save the combined results
%writetable(all_results, 'L:\rsmith\lab-members\cgoldman\Wellbeing\social_media\analysis\all_reaction_times_social_media_6-25-25.csv');
