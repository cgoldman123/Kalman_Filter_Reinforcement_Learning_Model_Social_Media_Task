
function make_plots_model_statistics(reward_diff_summary_table, choice_num_summary_table)
    % Set the study info
    study_info.num_games = 40;
    study_info.num_forced_choices = 4;
    study_info.num_free_choices_big_hor = 5;
    study_info.num_choices_big_hor = 9;
    

    % Figure out if plotting choices and RTs or just choices (for the
    % choice-only models). See if mean_rt_hor is a field of
    % choice_num_summary_table
    if any(contains(fieldnames(choice_num_summary_table), 'mean_rt_hor'))
        plot_rt_by_reward_diff(reward_diff_summary_table,study_info);
    end
    plot_total_uncert_and_estimated_rdiff(reward_diff_summary_table,study_info);
    plot_behavior_by_choice_num(choice_num_summary_table,study_info);
    plot_prob_choice_by_rdiff(reward_diff_summary_table,study_info); 

end