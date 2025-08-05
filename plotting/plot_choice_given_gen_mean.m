function plot_choice_given_gen_mean(root, fitting_procedure, experiment,room_type, results_dir, MDP, id, gen_mean_difference, horizon, truncate_big_hor)
    % Set the study info
    study_info.num_games = 40;
    study_info.num_forced_choices = 4;
    study_info.num_free_choices_big_hor = 5;
    study_info.num_choices_big_hor = 9;

    % First call get_fits to get the schedule/forced choices before
    MDP.get_processed_behavior_and_dont_fit_model = 1; % Toggle on to extract the rts and other processed behavioral data but not fit the model
    MDP.fit_model = 1; % Toggle on even though the model won't fit
    [rt_data, mdp] = get_fits(root, fitting_procedure, experiment,room_type, results_dir, MDP, id);
    
    %load('./SPM_scripts/social_media_prolific_mdp_cb1.mat'); 
    mdp_fieldnames = fieldnames(mdp);
    for (i=1:length(mdp_fieldnames))
        MDP.(mdp_fieldnames{i}) = mdp.(mdp_fieldnames{i});
    end
    params = MDP.params;
    model = MDP.model;
    % Locate games of interest based on gen_mean_difference and horizon
    if truncate_big_hor
        horizon = study_info.num_free_choices_big_hor;
        % run model unnecessarily to locate games of interest within big hor
        actions_and_rts.actions = mdp.actions;
        actions_and_rts.RTs = nan(study_info.num_games,study_info.num_choices_big_hor);
        model_output = model(params, actions_and_rts, mdp.rewards, MDP, 1);
        games_of_interest = locate_games_of_interest(mdp, model_output, gen_mean_difference, horizon);
        % run the model again treating every game like Small hor
        mdp.C1 = ones(1,study_info.num_games);
        actions_and_rts.actions(:, 6:study_info.num_choices_big_hor) = NaN;
        mdp.rewards(:, 6:study_info.num_choices_big_hor) = NaN;
        model_output = model(params, actions_and_rts, mdp.rewards, MDP, 1);

    else
        actions_and_rts.actions = mdp.actions;
        actions_and_rts.RTs = nan(study_info.num_games,study_info.num_choices_big_hor);
        model_output = model(params, actions_and_rts, mdp.rewards, MDP, 1);
        games_of_interest = locate_games_of_interest(mdp, model_output, gen_mean_difference, horizon);
    end

    
    
    % Call the model function to get model output
    
    
    
    % Plot the games of interest
    plot_bandit_games(model_output, games_of_interest,study_info);
end

function games_of_interest = locate_games_of_interest(mdp, model_output, gen_mean_difference, horizon)
    % Calculate the mean difference between the first 4 choices of each bandit
    mean_bandit1 = mean(mdp.bandit1_schedule(:, 1:4), 2);
    mean_bandit2 = mean(mdp.bandit2_schedule(:, 1:4), 2);
    mean_diff = abs(mean_bandit1 - mean_bandit2);

    % Define target values based on gen_mean_difference
    if gen_mean_difference == 24
        target_values = [23.7500, 24.0000, 24.2500];
    else
        target_values = [gen_mean_difference];
    end

    % Find the rows where the mean difference matches target values
    rows_with_gen_mean_diff = find(ismember(mean_diff, target_values));

    % Count the number of NaN values in each row for horizon filtering
    nan_counts = sum(isnan(model_output.actions), 2);
    if horizon == 1
        rows_with_horizon = find(nan_counts == 4);
    else
        rows_with_horizon = find(nan_counts == 0);
    end

    % Find the intersection of rows that match both criteria
    games_of_interest = intersect(rows_with_gen_mean_diff, rows_with_horizon);
end

function plot_bandit_games(model_output, games_of_interest,study_info)
    num_games = length(games_of_interest);
    num_choices = study_info.num_choices_big_hor;  % Each game has several total choices

    figure;
    


    % Loop through each game of interest and create the plots
    for game_idx = 1:num_games
        game = games_of_interest(game_idx);
        
        % Extract free choices and rewards for the current game
        free_choices = model_output.actions(game, :);
        rewards = model_output.rewards(game, :);
        action_probs = model_output.action_probs(game, :);
        action_probs(isnan(action_probs)) = 1;

        % Create a subplot for the game
        subplot(ceil(num_games/2), 2, game_idx);
        hold on;
        
        % Label the subplot as either Small hor or big hor based on the number of free choices
        if sum(~isnan(free_choices(5:end))) == 1
            title(['Small hor - Game ', num2str(game)]);
        else
            title(['Big hor - Game ', num2str(game)]);
        end
        
        % Plot the two columns representing the two bandits with several cells each
        for row_idx = 1:num_choices
            % Left bandit column (bandit 1)
            rectangle('Position', [1, num_choices+1-row_idx, 1, 1], 'EdgeColor', 'k');
            % Right bandit column (bandit 2)
            rectangle('Position', [3, num_choices+1-row_idx, 1, 1], 'EdgeColor', 'k');
        end
        
        % Loop over choices to place rewards in the correct bandit column
        for choice_idx = 1:num_choices
            if ~isnan(free_choices(choice_idx))
                % Determine the shading based on action probability (darker = closer to 1)
                prob_shading = 1 - action_probs(choice_idx);  % Higher prob = darker
                shading_color = [prob_shading, prob_shading, prob_shading];  % Grayscale
                not_chosen_color = [1 - prob_shading, 1 - prob_shading, 1 - prob_shading];
                % Adjust the y-coordinate for correct placement (subtract 1 from 'choice_idx')
                y_pos = num_choices+1 - choice_idx + 1;  % Corrected y position

                if free_choices(choice_idx) == 1
                    % Chose the left bandit, place reward and shading in the left column
                    fill([1, 2, 2, 1], [y_pos, y_pos, y_pos-1, y_pos-1], shading_color);
                    text(1.5, y_pos-0.5, num2str(rewards(choice_idx)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', [0, .7, 0]);  % Neon green text
                
                    % not chosen bandit
                    fill([3, 4, 4, 3], [y_pos, y_pos, y_pos-1, y_pos-1], not_chosen_color);

                elseif free_choices(choice_idx) == 2
                    % Chose the right bandit, place reward and shading in the right column
                    fill([3, 4, 4, 3], [y_pos, y_pos, y_pos-1, y_pos-1], shading_color);
                    text(3.5, y_pos-0.5, num2str(rewards(choice_idx)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', [0, .7, 0]);  % Neon green text
                
                    % not chosen bandit
                    fill([1, 2, 2, 1], [y_pos, y_pos, y_pos-1, y_pos-1], not_chosen_color);

                end
            end
        end

        
        % Format the plot
        axis([0 5 0 num_choices+1]);  % Set axis limits to fit two columns
        axis off;  % Turn off axis labels for cleaner plot
        hold off;
        

        % add color bar
        colormap(flipud(gray));

        % Create a colorbar for the entire figure
        c = colorbar('Location', 'eastoutside');  % Position the colorbar outside the subplots
        caxis([0 1]);  % Ensure the colorbar ranges from 0 (light) to 1 (dark)

    end

end
