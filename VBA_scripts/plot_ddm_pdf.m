function plot_ddm_pdf(drift, starting_bias, decision_thresh, min_time, max_time)
    % PLOT_DDM_PDF Plots the probability density function (PDF) of a
    % drift-diffusion model (DDM) using wfpt.
    %
    % Inputs:
    %   drift - Drift rate of the DDM
    %   starting_bias - Starting bias (initial value of the process)
    %   decision_thresh - Decision threshold (boundary value)
    %   min_time - Minimum time to plot (default = 0)
    %   max_time - Maximum time to plot (default = 5)
    %
    % Example usage:
    %   plot_ddm_pdf(0.5, 0.5, 1, 0, 5)

    % Set default values for min_time and max_time if not provided
    if nargin < 4
        min_time = 0;
    end
    if nargin < 5
        max_time = 5;
    end

    % Define time range for plotting
    time_range = linspace(min_time, max_time, 1000);

    % Preallocate array for PDF values
    pdf_values = zeros(size(time_range));

    % Compute PDF for each time point using wfpt
    for i = 1:length(time_range)
        pdf_values(i) = wfpt(time_range(i), drift, decision_thresh, starting_bias);
    end

    % Plot the PDF with interactive data cursor enabled
    figure;
    plot(time_range, pdf_values, 'LineWidth', 2);
    xlabel('Reaction Time (s)');
    ylabel('Probability Density');
    title(sprintf('Drift-Diffusion Model PDF\nDrift Rate: %.2f, Starting Bias: %.2f, Decision Threshold: %.2f', drift, starting_bias, decision_thresh));
    grid on;

    % Enable data cursor mode for interactive exploration
    datacursormode on;
end
