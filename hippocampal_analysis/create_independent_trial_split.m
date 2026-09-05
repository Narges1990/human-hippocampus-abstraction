function [classification_trials, analysis_trials, split_summary] = ...
    create_independent_trial_split(neu)

% ============================================================
% Creates a fixed 50/50 split of trials independently
% within each stimulus x context condition for every neuron.
%
% classification_trials:
%     used ONLY for category-like / identity-like classification
%
% analysis_trials:
%     used ONLY for pseudopopulation analyses
%
% This guarantees that the same trial is never used both
% for subtype classification and downstream CCGP/geometry.
% ============================================================


n_neurons = numel(neu.array);


classification_trials = ...
    cell(n_neurons,1);


analysis_trials = ...
    cell(n_neurons,1);



neuron_col = ...
    (1:n_neurons)';


n_classification = ...
    zeros(n_neurons,1);


n_analysis = ...
    zeros(n_neurons,1);


minimum_classification_per_condition = ...
    NaN(n_neurons,1);


minimum_analysis_per_condition = ...
    NaN(n_neurons,1);



for neuron_id = 1:n_neurons


    T = neu.array{neuron_id};


    class_idx_all = [];

    analysis_idx_all = [];


    class_counts = [];

    analysis_counts = [];



    for ctx = 1:2

        for stim = 1:4


            idx = find( ...
                T.context == ctx & ...
                T.stim_id == stim);


            n_this = ...
                numel(idx);


            if n_this < 2

                error( ...
                    ['Neuron %d, stimulus %d, context %d ' ...
                     'contains fewer than two trials.'], ...
                     neuron_id, ...
                     stim, ...
                     ctx);

            end



            % Randomize trials within this condition
            perm = ...
                randperm(n_this);


            idx = ...
                idx(perm);



            % Approximately 50/50
            n_class = ...
                floor(n_this/2);



            class_idx = ...
                idx(1:n_class);


            analysis_idx = ...
                idx(n_class+1:end);



            class_idx_all = [ ...
                class_idx_all; ...
                class_idx(:) ...
                ];


            analysis_idx_all = [ ...
                analysis_idx_all; ...
                analysis_idx(:) ...
                ];



            class_counts(end+1,1) = ...
                numel(class_idx);


            analysis_counts(end+1,1) = ...
                numel(analysis_idx);

        end

    end



    classification_trials{neuron_id} = ...
        class_idx_all;


    analysis_trials{neuron_id} = ...
        analysis_idx_all;



    n_classification(neuron_id) = ...
        numel(class_idx_all);


    n_analysis(neuron_id) = ...
        numel(analysis_idx_all);


    minimum_classification_per_condition(neuron_id) = ...
        min(class_counts);


    minimum_analysis_per_condition(neuron_id) = ...
        min(analysis_counts);

end



split_summary = table();


split_summary.neuron = ...
    neuron_col;


split_summary.n_classification = ...
    n_classification;


split_summary.n_analysis = ...
    n_analysis;


split_summary.min_classification_per_condition = ...
    minimum_classification_per_condition;


split_summary.min_analysis_per_condition = ...
    minimum_analysis_per_condition;


end