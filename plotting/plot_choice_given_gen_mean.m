function plot_choice_given_gen_mean(processed_data, MDP, gen_mean_difference, horizon, truncate_big_hor)
    
    MDP.processed_data = processed_data;
    params = MDP.params;
    model = MDP.model;
    num_choices_big_hor = processed_data.num_forced_choices+processed_data.num_free_choices_big_hor;
    % Locate games of interest based on gen_mean_difference and horizon
    if truncate_big_hor
        horizon = processed_data.num_free_choices_big_hor;
        % run model unnecessarily to locate games of interest within big hor
        actions_and_rts.actions = processed_data.actions;
        actions_and_rts.RTs = nan(processed_data.num_games,num_choices_big_hor);
        model_output = model(params, actions_and_rts, processed_data.rewards, MDP, 1);
        games_of_interest = locate_games_of_interest(processed_data, model_output, gen_mean_difference, horizon);
        % run the model again treating every game like Small hor
        processed_data.horizon_type = ones(processed_data.num_games,1);
        actions_and_rts.actions(:, 6:num_choices_big_hor) = NaN;
        processed_data.rewards(:, 6:num_choices_big_hor) = NaN;
        model_output = model(params, actions_and_rts, processed_data.rewards, MDP, 1);

    else
        actions_and_rts.actions = processed_data.actions;
        actions_and_rts.RTs = nan(processed_data.num_games,num_choices_big_hor);
        model_output = model(params, actions_and_rts, processed_data.rewards, MDP, 1);
        games_of_interest = locate_games_of_interest(processed_data, model_output, gen_mean_difference, horizon);
    end
    
    % Plot the games of interest
    plot_bandit_games(model_output, games_of_interest,processed_data);
end

function games_of_interest = locate_games_of_interest(processed_data, model_output, gen_mean_difference, horizon)
    % Calculate the mean difference between bandits
    if isfield(processed_data, 'bandit1_mean') && isfield(processed_data,'bandit2_mean')
        mean_bandit1 = processed_data.bandit1_mean; % get gen mean
        mean_bandit2 = processed_data.bandit2_mean; % get gen mean
    else
        mean_bandit1 = mean(processed_data.bandit1_schedule(:,1:4), 2); % get mean of forced choices on left
        mean_bandit2 = mean(processed_data.bandit2_schedule(:,1:4), 2); % get mean of forced choices on right
    end
    
    mean_diff = abs(mean_bandit1 - mean_bandit2);

    % Define target values based on gen_mean_difference
    if gen_mean_difference == 24
        target_values = [23.7500, 24.0000, 24.2500];
    else
        target_values = [gen_mean_difference];
    end

    % Find the rows where the mean difference matches target values
    rows_with_gen_mean_diff = find(ismember(mean_diff, target_values));

    % Find the rows with this horizon
    if horizon ==1
        rows_with_horizon = find(processed_data.horizon_type == 1);
    else
        rows_with_horizon = find(processed_data.horizon_type == 2);
    end

    % Find the intersection of rows that match both criteria
    games_of_interest = intersect(rows_with_gen_mean_diff, rows_with_horizon);
end

function plot_bandit_games(model_output, games_of_interest,processed_data)
    num_games = length(games_of_interest);
    num_choices = processed_data.num_free_choices_big_hor + processed_data.num_forced_choices;  % Each game has several total choices

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
