function [rt, acc] = simulate_DDM(drift, threshold, ndt, starting_bias, noise_constant, dt, max_rt)
% SIMULATE_DDM Simulates reaction time and accuracy for a single trial
% using a Drift Diffusion Model (DDM).
%
% Parameters
% ----------
% drift : double
%     Drift rate of the diffusion decision model.
% threshold : double
%     Decision threshold of the diffusion decision model.
% ndt : double
%     Non-decision time (seconds).
% starting_bias : double
%     Relative starting point of the diffusion process (0 to 1).
% noise_constant : double
%     Scaling factor for the noise in the diffusion process.
% dt : double
%     Time step for the simulation (seconds).
% max_rt : double
%     Maximum allowable response time (seconds).
%
% Returns
% -------
% rt : double
%     Simulated response time for the trial.
% acc : double
%     Simulated accuracy (1 for correct, 0 for incorrect).
%
% Example
% -------
% [rt, acc] = simulate_DDM(0.1, 1, 0.3, 0.5, 1, 0.001, 10);

    % Initialize variables
    x = starting_bias * threshold; % Initial evidence level
    tstep = 0; % Time step counter
    max_tsteps = ceil(max_rt / dt); % Maximum number of steps allowed

    % Simulate diffusion process
    while tstep < max_tsteps
        % Update evidence
        x = x + normrnd(drift * dt, noise_constant * sqrt(dt));
        tstep = tstep + 1;

        % Check for threshold crossings
        if x >= threshold
            acc = 1; % Correct response
            rt = tstep * dt + ndt; % Response time includes non-decision time
            return;
        elseif x <= 0
            acc = 0; % Incorrect response
            rt = tstep * dt + ndt; % Response time includes non-decision time
            return;
        end
    end

    % If max_rt is reached without crossing thresholds
    rt = max_rt;
    acc = NaN; % No decision made
end
