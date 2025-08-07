function make_plots_param_sweep(simmed_model_output,processed_data)
    processed_data.num_choices_big_hor = processed_data.num_free_choices_big_hor + processed_data.num_forced_choices;
    % Figure out if plotting choices and RTs or just choices (for the
    % choice-only models). See if mean_rt_hor is a field of
    % the first cell in simmed_model_output
    example_stats_table = simmed_model_output{1,3};
    if any(contains(fieldnames(example_stats_table), 'mean_rt_hor'))
        plot_sweep_rt(simmed_model_output,processed_data);
    end
    plot_sweep_accuracy(simmed_model_output,processed_data);
    plot_sweep_high_info(simmed_model_output,processed_data);
end