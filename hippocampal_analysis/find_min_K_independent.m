function [min_K, limiting_info] = find_min_K_independent(neu, analysis_trials)

% ============================================================
% Find the minimum number of available held-out analysis trials
% across ALL neurons and ALL 8 stimulus x context conditions.
%
% Output:
%   min_K         = maximum K that can safely be used for every
%                   neuron and every condition
%
%   limiting_info = information about the neuron/condition
%                   that determines min_K
% ============================================================

n_neurons = numel(neu.array);

min_K = inf;

limiting_neuron = NaN;
limiting_context = NaN;
limiting_stimulus = NaN;


for neuron_id = 1:n_neurons

    T = neu.array{neuron_id};

    % Held-out trials for this neuron
    heldout_idx = analysis_trials{neuron_id};


    for ctx = 1:2

        for stim = 1:4

            % Trials belonging to this condition
            condition_idx = find( ...
                T.context == ctx & ...
                T.stim_id == stim);

            % Only held-out analysis trials
            available_idx = intersect( ...
                condition_idx, ...
                heldout_idx);

            n_available = numel(available_idx);


            % Update global minimum
            if n_available < min_K

                min_K = n_available;

                limiting_neuron = neuron_id;
                limiting_context = ctx;
                limiting_stimulus = stim;

            end

        end

    end

end


% Information about limiting condition
limiting_info = table( ...
    limiting_neuron, ...
    limiting_context, ...
    limiting_stimulus, ...
    min_K, ...
    'VariableNames', { ...
    'neuron', ...
    'context', ...
    'stimulus', ...
    'n_trials'});


fprintf('\n============================================\n');
fprintf('MINIMUM HELD-OUT TRIAL COUNT\n');
fprintf('============================================\n');

fprintf('Minimum K = %d\n', min_K);

fprintf('Limiting neuron = %d\n', limiting_neuron);

fprintf('Context = %d\n', limiting_context);

fprintf('Stimulus = %d\n', limiting_stimulus);

fprintf('============================================\n\n');


end