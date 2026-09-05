function [X,labels] = ...
    build_pseudopopulation_independent( ...
    neu, ...
    neuron_ids, ...
    K, ...
    analysis_trials)

% ============================================================
% Builds pseudopopulations using ONLY independent held-out
% analysis trials.
%
% Rows:
%     8 conditions x K trials
%
% Columns:
%     neurons
%
% Condition order:
%     C1-S1
%     C1-S2
%     C1-S3
%     C1-S4
%     C2-S1
%     C2-S2
%     C2-S3
%     C2-S4
%
% Category:
%     stimulus 1 + 3 = category 1
%     stimulus 2 + 4 = category 2
% ============================================================


n_neurons = ...
    numel(neuron_ids);


n_conditions = 8;


n_rows = ...
    n_conditions*K;



X = ...
    NaN(n_rows,n_neurons);



stim_label = ...
    NaN(n_rows,1);


context_label = ...
    NaN(n_rows,1);


category_label = ...
    NaN(n_rows,1);



row_counter = 1;



for ctx = 1:2

    for stim = 1:4


        rows_this_condition = ...
            row_counter:(row_counter+K-1);



        stim_label(rows_this_condition) = ...
            stim;


        context_label(rows_this_condition) = ...
            ctx;



        if stim == 1 || stim == 3

            category_label(rows_this_condition) = ...
                1;

        else

            category_label(rows_this_condition) = ...
                2;

        end



        for n = 1:n_neurons


            neuron_id = ...
                neuron_ids(n);


            T = ...
                neu.array{neuron_id};



            % All trials belonging to this condition
            idx_condition = find( ...
                T.context == ctx & ...
                T.stim_id == stim);



            % Trials assigned to held-out analysis set
            idx_analysis = ...
                analysis_trials{neuron_id};



            % Intersection guarantees that no classification
            % trial can enter the population analysis.
            idx = intersect( ...
                idx_condition, ...
                idx_analysis);



            if numel(idx) < K

                X = [];

                labels = [];

                return

            end



            chosen = ...
                idx(randperm(numel(idx),K));



            X(rows_this_condition,n) = ...
                T.fr_stim(chosen);

        end



        row_counter = ...
            row_counter + K;

    end

end



labels = table();


labels.stim = ...
    stim_label;


labels.context = ...
    context_label;


labels.category = ...
    category_label;


end