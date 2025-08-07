function make_plots_param_sweep(simmed_model_output)
    % Set the study info
    study_info.num_games = 40;
    study_info.num_forced_choices = 4;
    study_info.num_free_choices_big_hor = 5;
    study_info.num_choices_big_hor = 9;
    % Figure out if plotting choices and RTs or just choices (for the
    % choice-only models). See if mean_rt_hor is a field of
    % the first cell in simmed_model_output
    example_stats_table = simmed_model_output{1,3};
    if any(contains(fieldnames(example_stats_table), 'mean_rt_hor'))
        plot_sweep_rt(simmed_model_output,study_info);
    end
    plot_sweep_accuracy(simmed_model_output,study_info);
    plot_sweep_high_info(simmed_model_output,study_info);
end