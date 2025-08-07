
function make_plots_model_statistics(reward_diff_summary_table, choice_num_summary_table,processed_data)
    processed_data.num_choices_big_hor = processed_data.num_free_choices_big_hor + processed_data.num_forced_choices;

    % Figure out if plotting choices and RTs or just choices (for the
    % choice-only models). See if mean_rt_hor is a field of
    % choice_num_summary_table
    if any(contains(fieldnames(choice_num_summary_table), 'mean_rt_hor'))
        plot_rt_by_reward_diff(reward_diff_summary_table,processed_data);
    end
    plot_total_uncert_and_estimated_rdiff(reward_diff_summary_table,processed_data);
    plot_behavior_by_choice_num(choice_num_summary_table,processed_data);
    plot_prob_choice_by_rdiff(reward_diff_summary_table,processed_data); 

end