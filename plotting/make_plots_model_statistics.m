
function make_plots_model_statistics(reward_diff_summary_table, choice_num_summary_table)
    plot_rt_by_reward_diff(reward_diff_summary_table);
    plot_total_uncert_and_estimated_rdiff(reward_diff_summary_table);
    plot_behavior_by_choice_num(choice_num_summary_table);
    plot_prob_choice_by_rdiff(reward_diff_summary_table);

    


end