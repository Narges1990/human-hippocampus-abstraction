%% ============================================================
%  HUMAN HIPPOCAMPUS ANALYSIS
%  Independent-trial classification and population analysis
%
%  IMPORTANT:
%  - Full-data encoding model identifies stimulus/context/response/reward neurons.
%  - Trials are then split 50/50 within each stimulus x context condition.
%  - Classification half is used ONLY to classify stimulus neurons as:
%       category-like vs identity-like
%  - Analysis half is used ONLY for:
%       pseudopopulations
%       CCGP
%       PCA geometry
%       category-axis alignment
%       regression
%       partial correlations
%       mediation
%       
%  - These analysis are done ONLY for  category-like: 
%       category-axis alignment
%       regression
%       partial correlations
%       mediation
% ============================================================

clearvars -except neu
clc

rng(1);


%% ============================================================
% 1. SINGLE-NEURON ENCODING ANALYSIS FOR HIPPOCAMPUS
% ============================================================

hpc_idx = find(neu.cellinfo == 3);
n_hpc = numel(hpc_idx);

encoding_results = table();


for ii = 1:n_hpc

    neuron_id = hpc_idx(ii);

    T = neu.array{neuron_id};

    % Clean table
    D = table();

    D.fr_stim  = T.fr_stim(:);
    D.stim_cat = categorical(T.stim_id(:));
    D.context  = T.context(:);
    D.response = T.response(:);
    D.reward   = T.reward(:);


    % Full encoding model
    mdl = fitlm(D, ...
        'fr_stim ~ stim_cat + context + response + reward');


    row = table();

    row.neuron = neuron_id;
    row.area = neu.cellinfo(neuron_id);
    row.sessionID = string(neu.sessionID{neuron_id});
    row.R2 = mdl.Rsquared.Ordinary;


    % Predictor p-values
    row.p_stim     = getPredictorP(mdl, 'stim_cat');
    row.p_context  = getPredictorP(mdl, 'context');
    row.p_response = getPredictorP(mdl, 'response');
    row.p_reward   = getPredictorP(mdl, 'reward');


    % Reduced models
    mdl_no_stim = fitlm(D, ...
        'fr_stim ~ context + response + reward');

    mdl_no_context = fitlm(D, ...
        'fr_stim ~ stim_cat + response + reward');

    mdl_no_response = fitlm(D, ...
        'fr_stim ~ stim_cat + context + reward');

    mdl_no_reward = fitlm(D, ...
        'fr_stim ~ stim_cat + context + response');


    % Delta R2
    row.deltaR2_stim = ...
        mdl.Rsquared.Ordinary - ...
        mdl_no_stim.Rsquared.Ordinary;

    row.deltaR2_context = ...
        mdl.Rsquared.Ordinary - ...
        mdl_no_context.Rsquared.Ordinary;

    row.deltaR2_response = ...
        mdl.Rsquared.Ordinary - ...
        mdl_no_response.Rsquared.Ordinary;

    row.deltaR2_reward = ...
        mdl.Rsquared.Ordinary - ...
        mdl_no_reward.Rsquared.Ordinary;


    encoding_results = [encoding_results; row];

end



%% ============================================================
% 2. CLASSIFY FUNCTIONAL NEURON TYPE
% ============================================================

n = height(encoding_results);

neuron_type = strings(n,1);
max_var = strings(n,1);

max_deltaR2 = NaN(n,1);
n_sig_vars = zeros(n,1);


for i = 1:n

    pvals = [ ...
        encoding_results.p_stim(i), ...
        encoding_results.p_context(i), ...
        encoding_results.p_response(i), ...
        encoding_results.p_reward(i) ...
        ];


    effects = [ ...
        encoding_results.deltaR2_stim(i), ...
        encoding_results.deltaR2_context(i), ...
        encoding_results.deltaR2_response(i), ...
        encoding_results.deltaR2_reward(i) ...
        ];


    var_names = [ ...
        "stimulus", ...
        "context", ...
        "response", ...
        "reward" ...
        ];


    sig_idx = pvals < 0.05;

    n_sig_vars(i) = sum(sig_idx);


    if n_sig_vars(i) == 0

        neuron_type(i) = "non_selective";

        max_var(i) = "none";

        max_deltaR2(i) = NaN;


    elseif n_sig_vars(i) == 1

        neuron_type(i) = var_names(sig_idx);

        max_var(i) = var_names(sig_idx);

        max_deltaR2(i) = effects(sig_idx);


    else

        sig_effects = effects(sig_idx);

        sig_vars = var_names(sig_idx);


        [max_val, max_idx] = max(sig_effects);


        neuron_type(i) = ...
            sig_vars(max_idx) + "_dominant";

        max_var(i) = ...
            sig_vars(max_idx);

        max_deltaR2(i) = max_val;

    end

end


encoding_results.neuron_type = neuron_type;
encoding_results.max_var = max_var;
encoding_results.max_deltaR2 = max_deltaR2;
encoding_results.n_sig_vars = n_sig_vars;


hpc_encoding = encoding_results;



fprintf('\n====================================================\n');
fprintf('HIPPOCAMPAL FUNCTIONAL CLASSIFICATION\n');
fprintf('====================================================\n');

disp('Hippocampal neuron type counts:');
tabulate(hpc_encoding.neuron_type)



%% ============================================================
% 3. MERGE DOMINANT AND SINGLE-VARIABLE TYPES
% ============================================================

merged_type = strings(height(hpc_encoding),1);


for i = 1:height(hpc_encoding)

    t = hpc_encoding.neuron_type(i);


    if contains(t,"stimulus")

        merged_type(i) = "stimulus";

    elseif contains(t,"context")

        merged_type(i) = "context";

    elseif contains(t,"response")

        merged_type(i) = "response";

    elseif contains(t,"reward")

        merged_type(i) = "reward";

    else

        merged_type(i) = "non_selective";

    end

end


hpc_encoding.merged_type = merged_type;


disp('Merged hippocampal neuron types:');
tabulate(hpc_encoding.merged_type)



%% ============================================================
% 4. CREATE FIXED INDEPENDENT TRIAL SPLIT
%
% Classification trials:
%   ONLY category-like / identity-like classification
%
% ============================================================

fprintf('\n====================================================\n');
fprintf('CREATING INDEPENDENT TRIAL SPLIT\n');
fprintf('====================================================\n');


rng(1001);


[classification_trials, analysis_trials, split_summary] = ...
    create_independent_trial_split(neu);


disp('Trial split summary:');
disp(split_summary);


fprintf('\nMean classification trials/neuron: %.1f\n', ...
    mean(split_summary.n_classification));

fprintf('Mean analysis trials/neuron: %.1f\n', ...
    mean(split_summary.n_analysis));


[min_K, limiting_info] = ...
    find_min_K_independent(neu, analysis_trials);

disp(limiting_info);


%% ============================================================
% 5. SPLIT STIMULUS NEURONS INTO
%    CATEGORY-LIKE AND IDENTITY-LIKE
%
%    CRITICAL:
%    ONLY CLASSIFICATION TRIALS ARE USED HERE.
% ============================================================

stim_idx = find( ...
    hpc_encoding.merged_type == "stimulus");


stimulus_subtype = strings(numel(stim_idx),1);

category_score_all = ...
    NaN(numel(stim_idx),1);

identity_score_all = ...
    NaN(numel(stim_idx),1);



% Category template
cat_template = [ ...
     1 -1  1 -1 ...
     1 -1  1 -1 ...
     ]';


for k = 1:numel(stim_idx)

    neuron_id = ...
        hpc_encoding.neuron(stim_idx(k));


    T = neu.array{neuron_id};


    % ONLY classification-half trials
    class_idx = ...
        classification_trials{neuron_id};


    % 8-condition firing vector
    F = NaN(8,1);


    cnt = 1;


    for ctx = 1:2

        for stim = 1:4


            idx_condition = find( ...
                T.context == ctx & ...
                T.stim_id == stim);


            % Keep ONLY classification trials
            idx = intersect( ...
                idx_condition, ...
                class_idx);


            if isempty(idx)

                error( ...
                    ['Neuron %d has no classification trials ' ...
                     'for stimulus %d context %d.'], ...
                     neuron_id, stim, ctx);

            end


            F(cnt) = ...
                mean(T.fr_stim(idx), 'omitnan');


            cnt = cnt + 1;

        end

    end



    % z-score across eight conditions
    F = (F - mean(F,'omitnan')) ./ ...
        (std(F,'omitnan') + eps);



    % CATEGORY SCORE

    r_cat = corr( ...
        F, ...
        cat_template, ...
        'rows','complete');


    category_score = abs(r_cat);



    % IDENTITY SCORE

    identity_scores = ...
        NaN(4,1);


    for stim = 1:4


        tmp = -ones(8,1);


        tmp(stim) = 1;

        tmp(stim+4) = 1;


        identity_scores(stim) = ...
            abs(corr( ...
                F, ...
                tmp, ...
                'rows','complete'));

    end


    identity_score = ...
        max(identity_scores);



    category_score_all(k) = ...
        category_score;

    identity_score_all(k) = ...
        identity_score;



    % CLASSIFICATION

    if category_score > identity_score

        stimulus_subtype(k) = ...
            "category_like";

    else

        stimulus_subtype(k) = ...
            "identity_like";

    end

end



stimulus_neuron_table = table();


stimulus_neuron_table.neuron = ...
    hpc_encoding.neuron(stim_idx);


stimulus_neuron_table.sessionID = ...
    hpc_encoding.sessionID(stim_idx);


stimulus_neuron_table.category_score = ...
    category_score_all;


stimulus_neuron_table.identity_score = ...
    identity_score_all;


stimulus_neuron_table.subtype = ...
    stimulus_subtype;



fprintf('\n====================================================\n');
fprintf('INDEPENDENT-TRIAL STIMULUS SUBTYPE CLASSIFICATION\n');
fprintf('====================================================\n');


tabulate(stimulus_neuron_table.subtype)



cat_only = stimulus_neuron_table( ...
    stimulus_neuron_table.subtype == "category_like",:);


cat_only = sortrows( ...
    cat_only, ...
    'category_score', ...
    'descend');


disp('Top category-like neurons:');

disp( ...
    cat_only( ...
    1:min(20,height(cat_only)), ...
    :) ...
    );



%% ============================================================
% 6. DEFINE NEURON POOLS
% ============================================================

identity_neurons = ...
    stimulus_neuron_table.neuron( ...
    stimulus_neuron_table.subtype == ...
    "identity_like");


category_neurons = ...
    stimulus_neuron_table.neuron( ...
    stimulus_neuron_table.subtype == ...
    "category_like");


context_neurons = ...
    hpc_encoding.neuron( ...
    hpc_encoding.merged_type == ...
    "context");


response_neurons = ...
    hpc_encoding.neuron( ...
    hpc_encoding.merged_type == ...
    "response");


reward_neurons = ...
    hpc_encoding.neuron( ...
    hpc_encoding.merged_type == ...
    "reward");


noise_neurons = ...
    hpc_encoding.neuron( ...
    hpc_encoding.merged_type == ...
    "non_selective");



fprintf('\nAvailable neuron pools:\n');

fprintf('Identity-like: %d\n', ...
    numel(identity_neurons));

fprintf('Category-like: %d\n', ...
    numel(category_neurons));

fprintf('Context: %d\n', ...
    numel(context_neurons));

fprintf('Response: %d\n', ...
    numel(response_neurons));

fprintf('Reward: %d\n', ...
    numel(reward_neurons));

fprintf('Non-selective: %d\n', ...
    numel(noise_neurons));


%% ============================================================
% 7. REAL-DATA SCENARIOS
%    ENCODING + CCGP
%
%    All X values come ONLY from held-out ANALYSIS trials.
% ============================================================

rng(1);


scenario_defs = table();


scenario_defs.name = [
    "BASE_ID13"
    "ID13_CAT5"
    "ID13_CAT10"
    "ID13_CAT13"
];


scenario_defs.n_identity = ...
    [13;13;13;13];


scenario_defs.n_category = ...
    [0;5;10;13];


scenario_defs.n_context = ...
    [56;56;56;56];


scenario_defs.n_response = ...
    [13;13;13;13];


scenario_defs.n_reward = ...
    [13;13;13;13];


scenario_defs.n_noise = ...
    [100;100;100;100];



n_iterations = 100;

K_trials = min_K;

scenario_results = table();



for s = 1:height(scenario_defs)


    fprintf('\nRunning scenario: %s\n', ...
        scenario_defs.name(s));


    for it = 1:n_iterations


        selected_identity = ...
            sample_neurons( ...
            identity_neurons, ...
            scenario_defs.n_identity(s));


        selected_category = ...
            sample_neurons( ...
            category_neurons, ...
            scenario_defs.n_category(s));


        selected_context = ...
            sample_neurons( ...
            context_neurons, ...
            scenario_defs.n_context(s));


        selected_response = ...
            sample_neurons( ...
            response_neurons, ...
            scenario_defs.n_response(s));


        selected_reward = ...
            sample_neurons( ...
            reward_neurons, ...
            scenario_defs.n_reward(s));


        selected_noise = ...
            sample_neurons( ...
            noise_neurons, ...
            scenario_defs.n_noise(s));



        selected_neurons = unique([ ...
            selected_identity
            selected_category
            selected_context
            selected_response
            selected_reward
            selected_noise ...
            ]);



        % CRITICAL:
        % ONLY held-out analysis trials enter X.
        [X,labels] = ...
            build_pseudopopulation_independent( ...
            neu, ...
            selected_neurons, ...
            K_trials, ...
            analysis_trials);



        if isempty(X)

            fprintf( ...
                'Skipped iteration %d: insufficient held-out trials.\n', ...
                it);

            continue

        end



        % Encoding measures retained from original encoding table
        enc = scenario_encoding_summary( ...
            hpc_encoding, ...
            selected_neurons);



        cat_ccgp = ...
            compute_category_CCGP( ...
            X,labels);


        ctx_ccgp = ...
            compute_context_CCGP( ...
            X,labels);



        row = table();


        row.scenario = ...
            scenario_defs.name(s);

        row.iteration = ...
            it;


        row.n_identity = ...
            numel(selected_identity);

        row.n_category = ...
            numel(selected_category);

        row.n_context = ...
            numel(selected_context);

        row.n_response = ...
            numel(selected_response);

        row.n_reward = ...
            numel(selected_reward);

        row.n_noise = ...
            numel(selected_noise);

        row.n_total = ...
            numel(selected_neurons);



        row.encoding_stimulus = ...
            enc.stimulus;

        row.encoding_context = ...
            enc.context;

        row.encoding_response = ...
            enc.response;

        row.encoding_reward = ...
            enc.reward;


        row.category_CCGP = ...
            cat_ccgp;

        row.context_CCGP = ...
            ctx_ccgp;



        scenario_results = ...
            [scenario_results;row];

    end

end



%% ============================================================
% 8. SCENARIO SUMMARY
% ============================================================

summary_results = groupsummary( ...
    scenario_results, ...
    "scenario", ...
    "mean", ...
    { ...
    'n_identity', ...
    'n_category', ...
    'n_context', ...
    'n_response', ...
    'n_reward', ...
    'n_noise', ...
    'n_total', ...
    'encoding_stimulus', ...
    'encoding_context', ...
    'encoding_response', ...
    'encoding_reward', ...
    'category_CCGP', ...
    'context_CCGP' ...
    });



desired_order = [
    "BASE_ID13"
    "ID13_CAT5"
    "ID13_CAT10"
    "ID13_CAT13"
];


summary_results.scenario = ...
    string(summary_results.scenario);


[~,order_idx] = ...
    ismember( ...
    desired_order, ...
    summary_results.scenario);


summary_results = ...
    summary_results(order_idx,:);


disp('Scenario summary:');

disp(summary_results);



%% ============================================================
% 9. CCGP FIGURE
% ============================================================

figure;


plot( ...
    summary_results.mean_category_CCGP, ...
    '-o', ...
    'LineWidth',2);


hold on;


yline(0.5,'--','Chance');


xticks(1:height(summary_results));


xticklabels({ ...
    'Base'
    '+5 CA'
    '+10 CA'
    '+13 CA' ...
    });


ylabel('AC vs BD CCGP');


title( ...
    'Held-out trials: Category CCGP');


grid on;
ax = gca;


%% ============================================================
% 10. ENCODING FIGURE
% ============================================================

figure;


plot( ...
    summary_results.mean_encoding_stimulus, ...
    '-o', ...
    'LineWidth',2);

hold on;



xticks(1:height(summary_results));


xticklabels({ ...
    'Base'
    '+5 CA'
    '+10 CA'
    '+13 CA' ...
    });


ylabel('Mean \DeltaR^2');


legend( ...
    'Stimulus', ...
    'Context', ...
    'Response', ...
    'Reward', ...
    'Location','best');


title('Encoding across scenarios');


grid on;

%% ============================================================
% 11. PCA / REPRESENTATIONAL GEOMETRY
%
% ONLY HELD-OUT ANALYSIS TRIALS ENTER THE GEOMETRY.
% ============================================================

rng(2);


n_geom_iterations = 100;

K_trials_geom = min_K;


geometry_results = table();


figure;

tiledlayout( ...
    1, ...
    height(scenario_defs), ...
    'TileSpacing','compact');



for s = 1:height(scenario_defs)


    scenario_name = ...
        scenario_defs.name(s);


    fprintf( ...
        '\nGeometry for scenario: %s\n', ...
        scenario_name);


    cond_mean_all = [];



    for it = 1:n_geom_iterations


        selected_identity = ...
            sample_neurons( ...
            identity_neurons, ...
            scenario_defs.n_identity(s));


        selected_category = ...
            sample_neurons( ...
            category_neurons, ...
            scenario_defs.n_category(s));


        selected_context = ...
            sample_neurons( ...
            context_neurons, ...
            scenario_defs.n_context(s));


        selected_response = ...
            sample_neurons( ...
            response_neurons, ...
            scenario_defs.n_response(s));


        selected_reward = ...
            sample_neurons( ...
            reward_neurons, ...
            scenario_defs.n_reward(s));


        selected_noise = ...
            sample_neurons( ...
            noise_neurons, ...
            scenario_defs.n_noise(s));



        selected_neurons = unique([ ...
            selected_identity
            selected_category
            selected_context
            selected_response
            selected_reward
            selected_noise ...
            ]);



        [X,labels] = ...
            build_pseudopopulation_independent( ...
            neu, ...
            selected_neurons, ...
            K_trials_geom, ...
            analysis_trials);



        if isempty(X)

            continue

        end



        C = ...
            compute_condition_means( ...
            X,labels);


        C = zscore(C,0,1);


        cond_mean_all(:,:,end+1) = C;

    end



    if isempty(cond_mean_all)

        error( ...
            'No valid geometry iterations for scenario %s.', ...
            scenario_name);

    end



    C_mean = ...
        mean( ...
        cond_mean_all, ...
        3, ...
        'omitnan');



    [~,score,~,~,explained] = ...
        pca(C_mean);



    score2 = ...
        score(:,1:2);



    stim_vec = ...
        [1 2 3 4 1 2 3 4]';


    ctx_vec = ...
        [1 1 1 1 2 2 2 2]';


    cat_vec = ...
        [1 2 1 2 1 2 1 2]';



    % Category separation
    c1 = mean( ...
        score2(cat_vec==1,:), ...
        1);


    c2 = mean( ...
        score2(cat_vec==2,:), ...
        1);


    category_sep = ...
        norm(c1-c2);



    % Context separation
    x1 = mean( ...
        score2(ctx_vec==1,:), ...
        1);


    x2 = mean( ...
        score2(ctx_vec==2,:), ...
        1);


    context_sep = ...
        norm(x1-x2);



    % Stimulus spread
    stimulus_spread = mean( ...
        vecnorm( ...
        score2 - mean(score2,1), ...
        2, ...
        2));



    row = table();


    row.scenario = ...
        scenario_name;

    row.PC1_var = ...
        explained(1);

    row.PC2_var = ...
        explained(2);

    row.category_sep = ...
        category_sep;

    row.context_sep = ...
        context_sep;

    row.stimulus_spread = ...
        stimulus_spread;


    geometry_results = ...
        [geometry_results;row];



    % Plot panel
    nexttile;

    hold on;



    for i = 1:8


        if cat_vec(i) == 1

            marker_face = ...
                [0.2 0.2 0.2];

        else

            marker_face = ...
                [0.8 0.8 0.8];

        end



        if ctx_vec(i) == 1

            mk = 'o';

        else

            mk = 's';

        end



        scatter( ...
            score2(i,1), ...
            score2(i,2), ...
            90, ...
            'Marker',mk, ...
            'MarkerFaceColor',marker_face, ...
            'MarkerEdgeColor','k');



        text( ...
            score2(i,1), ...
            score2(i,2), ...
            sprintf( ...
            'S%dC%d', ...
            stim_vec(i), ...
            ctx_vec(i)), ...
            'FontSize',8, ...
            'VerticalAlignment','bottom', ...
            'HorizontalAlignment','right');

    end



    xlabel( ...
        sprintf( ...
        'PC1 %.1f%%', ...
        explained(1)));


    ylabel( ...
        sprintf( ...
        'PC2 %.1f%%', ...
        explained(2)));


    title( ...
        strrep( ...
        string(scenario_name), ...
        '_','\_'));


    axis equal;

    grid on;

end


sgtitle( ...
    'Held-out-trial population geometry');


disp('Geometry summary:');

disp(geometry_results);



%% ============================================================
% 12. CATEGORY-AXIS ALIGNMENT ANALYSIS
% ============================================================

rng(3);


alignment_results = table();



for s = 1:height(scenario_defs)


    fprintf( ...
        '\nAxis alignment for scenario: %s\n', ...
        scenario_defs.name(s));


    for it = 1:n_iterations


        selected_identity = ...
            sample_neurons( ...
            identity_neurons, ...
            scenario_defs.n_identity(s));


        selected_category = ...
            sample_neurons( ...
            category_neurons, ...
            scenario_defs.n_category(s));


        selected_context = ...
            sample_neurons( ...
            context_neurons, ...
            scenario_defs.n_context(s));


        selected_response = ...
            sample_neurons( ...
            response_neurons, ...
            scenario_defs.n_response(s));


        selected_reward = ...
            sample_neurons( ...
            reward_neurons, ...
            scenario_defs.n_reward(s));


        selected_noise = ...
            sample_neurons( ...
            noise_neurons, ...
            scenario_defs.n_noise(s));



        selected_neurons = unique([ ...
            selected_identity
            selected_category
            selected_context
            selected_response
            selected_reward
            selected_noise ...
            ]);



        [X,labels] = ...
            build_pseudopopulation_independent( ...
            neu, ...
            selected_neurons, ...
            K_trials, ...
            analysis_trials);



        if isempty(X)

            continue

        end



        cat_ccgp = ...
            compute_category_CCGP( ...
            X,labels);


        ctx_ccgp = ...
            compute_context_CCGP( ...
            X,labels);


        geom = ...
            compute_category_axis_alignment( ...
            X,labels);



        row = table();


        row.scenario = ...
            scenario_defs.name(s);


        row.iteration = ...
            it;


        row.n_identity = ...
            numel(selected_identity);


        row.n_category = ...
            numel(selected_category);


        row.category_CCGP = ...
            cat_ccgp;


        row.context_CCGP = ...
            ctx_ccgp;


        row.category_axis_alignment = ...
            geom.category_axis_alignment;


        row.context_axis_alignment = ...
            geom.context_axis_alignment;


        row.category_axis_strength_ctx1 = ...
            geom.category_axis_strength_ctx1;


        row.category_axis_strength_ctx2 = ...
            geom.category_axis_strength_ctx2;


        row.mean_category_axis_strength = ...
            geom.mean_category_axis_strength;


        row.context_axis_strength_stim13 = ...
            geom.context_axis_strength_stim13;


        row.context_axis_strength_stim24 = ...
            geom.context_axis_strength_stim24;


        row.mean_context_axis_strength = ...
            geom.mean_context_axis_strength;



        alignment_results = ...
            [alignment_results;row];

    end

end



%% ============================================================
% 13. ALIGNMENT SUMMARY
% ============================================================

alignment_summary = groupsummary( ...
    alignment_results, ...
    "scenario", ...
    "mean", ...
    { ...
    'category_CCGP', ...
    'context_CCGP', ...
    'category_axis_alignment', ...
    'context_axis_alignment', ...
    'mean_category_axis_strength', ...
    'mean_context_axis_strength', ...
    'n_identity', ...
    'n_category' ...
    });



alignment_summary.scenario = ...
    string(alignment_summary.scenario);


[~,order_idx] = ...
    ismember( ...
    desired_order, ...
    alignment_summary.scenario);


alignment_summary = ...
    alignment_summary(order_idx,:);



disp('Alignment summary:');

disp(alignment_summary);





%% ============================================================
% 14. CORRELATIONS
% ============================================================

[r_align,p_align] = corr( ...
    alignment_results.category_axis_alignment, ...
    alignment_results.category_CCGP, ...
    'rows','complete');



[r_strength,p_strength] = corr( ...
    alignment_results.mean_category_axis_strength, ...
    alignment_results.category_CCGP, ...
    'rows','complete');



fprintf('\n====================================================\n');
fprintf('CORRELATION WITH HELD-OUT CATEGORY CCGP\n');
fprintf('====================================================\n');


fprintf( ...
    'Category-axis alignment: r = %.3f, p = %.5f\n', ...
    r_align, ...
    p_align);


fprintf( ...
    'Category-axis strength: r = %.3f, p = %.5f\n', ...
    r_strength, ...
    p_strength);



%% ============================================================
% 15. SENSITIVITY ANALYSIS:
%     SCENARIO-LEVEL CORRELATIONS
%
% Purpose:
% Test whether the alignment-CCGP and strength-CCGP relationships
% remain when the 100 resampling iterations within each neuronal
% composition are collapsed to a single scenario-level observation.
%
% Each scenario is defined by:
%     n_category x n_identity
%
% IMPORTANT:
% This is a robustness / sensitivity analysis.
% It does NOT replace the resample-level correlations or
% partial correlations above.
% ============================================================


fprintf('\n====================================================\n');
fprintf('SENSITIVITY ANALYSIS: SCENARIO-LEVEL CORRELATIONS\n');
fprintf('====================================================\n');


% 15.1 Aggregate the 100 resamples within each neuronal composition

scenario_sensitivity = groupsummary( ...
    alignment_results, ...
    {'n_category','n_identity'}, ...
    'mean', ...
    { ...
    'category_axis_alignment', ...
    'mean_category_axis_strength', ...
    'category_CCGP' ...
    });


% Rename variables for easier interpretation

scenario_sensitivity.Properties.VariableNames{ ...
    strcmp( ...
    scenario_sensitivity.Properties.VariableNames, ...
    'mean_category_axis_alignment')} = ...
    'mean_alignment';


scenario_sensitivity.Properties.VariableNames{ ...
    strcmp( ...
    scenario_sensitivity.Properties.VariableNames, ...
    'mean_mean_category_axis_strength')} = ...
    'mean_strength';


scenario_sensitivity.Properties.VariableNames{ ...
    strcmp( ...
    scenario_sensitivity.Properties.VariableNames, ...
    'mean_category_CCGP')} = ...
    'mean_CCGP';


% 15.2 Sort scenarios by neuronal composition

scenario_sensitivity = sortrows( ...
    scenario_sensitivity, ...
    {'n_identity','n_category'});


% 15.3 Create readable scenario labels

scenario_sensitivity.scenario = ...
    "ID" + string(scenario_sensitivity.n_identity) + ...
    "_CAT" + string(scenario_sensitivity.n_category);


% Move scenario label to first column

scenario_sensitivity = movevars( ...
    scenario_sensitivity, ...
    'scenario', ...
    'Before', ...
    1);


% 15.4 Scenario-level Pearson correlations

[r_align_scenario, p_align_scenario] = corr( ...
    scenario_sensitivity.mean_alignment, ...
    scenario_sensitivity.mean_CCGP, ...
    'Type','Pearson', ...
    'Rows','complete');


[r_strength_scenario, p_strength_scenario] = corr( ...
    scenario_sensitivity.mean_strength, ...
    scenario_sensitivity.mean_CCGP, ...
    'Type','Pearson', ...
    'Rows','complete');



% 15.5 Display scenario-level table

fprintf('\nScenario-level summary:\n');

disp( ...
    scenario_sensitivity(:, ...
    { ...
    'scenario', ...
    'n_category', ...
    'n_identity', ...
    'GroupCount', ...
    'mean_alignment', ...
    'mean_strength', ...
    'mean_CCGP' ...
    }) ...
    );


% 15.6 Direct comparison with original resample-level results

fprintf('\n====================================================\n');
fprintf('RESAMPLE-LEVEL vs SCENARIO-LEVEL CORRELATIONS\n');
fprintf('====================================================\n');


fprintf('\nALIGNMENT vs CCGP\n');

fprintf( ...
    'Resample level:  r = %.3f, p = %.5g\n', ...
    r_align, ...
    p_align);

fprintf( ...
    'Scenario level:  r = %.3f, p = %.5g\n', ...
    r_align_scenario, ...
    p_align_scenario);


fprintf('\nSTRENGTH vs CCGP\n');

fprintf( ...
    'Resample level:  r = %.3f, p = %.5g\n', ...
    r_strength, ...
    p_strength);

fprintf( ...
    'Scenario level:  r = %.3f, p = %.5g\n', ...
    r_strength_scenario, ...
    p_strength_scenario);


fprintf('\nNumber of independent neuronal compositions = %d\n', ...
    height(scenario_sensitivity));


% 15.7 Compact comparison table


sensitivity_comparison = table( ...
    ["Alignment"; "Strength"], ...
    [r_align; r_strength], ...
    [p_align; p_strength], ...
    [r_align_scenario; r_strength_scenario], ...
    [p_align_scenario; p_strength_scenario], ...
    'VariableNames', ...
    { ...
    'Measure', ...
    'r_resample', ...
    'p_resample', ...
    'r_scenario', ...
    'p_scenario' ...
    });


fprintf('\nCorrelation comparison table:\n');

disp(sensitivity_comparison);


% 15.8 Scenario-level sensitivity plots

figure;


subplot(1,2,1);

scatter( ...
    scenario_sensitivity.mean_alignment, ...
    scenario_sensitivity.mean_CCGP, ...
    80, ...
    'filled');

hold on;

lsline;

xlabel('Mean category-axis alignment');

ylabel('Mean held-out CCGP');

title( ...
    sprintf( ...
    'Scenario level: Alignment, r = %.2f, p = %.3g', ...
    r_align_scenario, ...
    p_align_scenario));

grid on;


subplot(1,2,2);

scatter( ...
    scenario_sensitivity.mean_strength, ...
    scenario_sensitivity.mean_CCGP, ...
    80, ...
    'filled');

hold on;

lsline;

xlabel('Mean category-axis strength');

ylabel('Mean held-out CCGP');

title( ...
    sprintf( ...
    'Scenario level: Strength, r = %.2f, p = %.3g', ...
    r_strength_scenario, ...
    p_strength_scenario));

grid on;


sgtitle( ...
    'Sensitivity analysis: correlations across neuronal compositions');







%% ============================================================
% 16. PARTIAL CORRELATIONS
% ============================================================

[r_align_partial,p_align_partial] = ...
    partialcorr( ...
    alignment_results.category_axis_alignment, ...
    alignment_results.category_CCGP, ...
    [ ...
    alignment_results.mean_category_axis_strength, ...
    alignment_results.n_identity, ...
    alignment_results.n_category ...
    ], ...
    'rows','complete');


[r_strength_partial,p_strength_partial] = ...
    partialcorr( ...
    alignment_results.mean_category_axis_strength, ...
    alignment_results.category_CCGP, ...
    [ ...
    alignment_results.category_axis_alignment, ...
    alignment_results.n_identity, ...
    alignment_results.n_category ...
    ], ...
    'rows','complete');


% FDR correction for partial correlations

p_partial = [
    p_align_partial;
    p_strength_partial
];

p_partial_FDR = mafdr(p_partial,'BHFDR',true);


fprintf('\nPartial correlations (FDR corrected):\n');

fprintf( ...
    'Alignment partial r = %.3f, FDR-corrected p = %.5f\n', ...
    r_align_partial, ...
    p_partial_FDR(1));


fprintf( ...
    'Strength partial r = %.3f, FDR-corrected p = %.5f\n', ...
    r_strength_partial, ...
    p_partial_FDR(2));

%% ============================================================
% 17. SCATTER PLOTS
% ============================================================

figure;


subplot(1,2,1);


scatter( ...
    alignment_results.category_axis_alignment, ...
    alignment_results.category_CCGP, ...
    40, ...
    'filled');


lsline;


xlabel('Category-axis alignment');

ylabel('Held-out CCGP');


title( ...
    sprintf( ...
    'Alignment: r = %.2f, p = %.3g', ...
    r_align, ...
    p_align));


grid on;



subplot(1,2,2);


scatter( ...
    alignment_results.mean_category_axis_strength, ...
    alignment_results.category_CCGP, ...
    40, ...
    'filled');


lsline;


xlabel('Category-axis strength');

ylabel('Held-out CCGP');


title( ...
    sprintf( ...
    'Strength: r = %.2f, p = %.3g', ...
    r_strength, ...
    p_strength));


grid on;



%% ============================================================
% 19. MEDIATION ANALYSIS
%
% category-like neuron number
%          ->
% category-axis alignment
%          ->
% held-out category CCGP
% ============================================================

M = alignment_results;



X_cat = ...
    M.n_category;


Med_align = ...
    M.category_axis_alignment;


Y_ccgp = ...
    M.category_CCGP;


Cov = [ ...
    M.n_identity, ...
    M.mean_category_axis_strength ...
    ];



%% Path a

tbl_a = table( ...
    X_cat, ...
    Cov(:,1), ...
    Cov(:,2), ...
    Med_align, ...
    'VariableNames', ...
    { ...
    'n_category', ...
    'n_identity', ...
    'axis_strength', ...
    'alignment' ...
    });



mdl_a = fitlm( ...
    tbl_a, ...
    'alignment ~ n_category + n_identity + axis_strength');


disp('Path a: n_category -> category-axis alignment');

disp(mdl_a);



a = mdl_a.Coefficients.Estimate( ...
    strcmp( ...
    mdl_a.CoefficientNames, ...
    'n_category'));



%% Path b and direct effect c'

tbl_b = table( ...
    X_cat, ...
    Cov(:,1), ...
    Cov(:,2), ...
    Med_align, ...
    Y_ccgp, ...
    'VariableNames', ...
    { ...
    'n_category', ...
    'n_identity', ...
    'axis_strength', ...
    'alignment', ...
    'CCGP' ...
    });



mdl_b = fitlm( ...
    tbl_b, ...
    ['CCGP ~ alignment + n_category + ' ...
     'n_identity + axis_strength']);


disp('Path b and c-prime:');

disp(mdl_b);



b = mdl_b.Coefficients.Estimate( ...
    strcmp( ...
    mdl_b.CoefficientNames, ...
    'alignment'));


c_prime = mdl_b.Coefficients.Estimate( ...
    strcmp( ...
    mdl_b.CoefficientNames, ...
    'n_category'));



%% Total effect c

tbl_c = table( ...
    X_cat, ...
    Cov(:,1), ...
    Cov(:,2), ...
    Y_ccgp, ...
    'VariableNames', ...
    { ...
    'n_category', ...
    'n_identity', ...
    'axis_strength', ...
    'CCGP' ...
    });



mdl_c = fitlm( ...
    tbl_c, ...
    'CCGP ~ n_category + n_identity + axis_strength');


disp('Total effect c:');

disp(mdl_c);



c_total = mdl_c.Coefficients.Estimate( ...
    strcmp( ...
    mdl_c.CoefficientNames, ...
    'n_category'));



indirect_effect = ...
    a*b;



fprintf('\n====================================================\n');
fprintf('MEDIATION SUMMARY\n');
fprintf('====================================================\n');


fprintf('Path a = %.5f\n',a);

fprintf('Path b = %.5f\n',b);

fprintf('Indirect effect a*b = %.5f\n', ...
    indirect_effect);

fprintf('Direct effect c'' = %.5f\n', ...
    c_prime);

fprintf('Total effect c = %.5f\n', ...
    c_total);



%% ============================================================
% 19. BOOTSTRAP MEDIATION
% ============================================================

rng(4);


n_boot = 5000;


boot_ab = ...
    NaN(n_boot,1);


N = height(M);



for ib = 1:n_boot


    boot_idx = ...
        randsample( ...
        N,N,true);


    Mb = ...
        M(boot_idx,:);



    Xb = ...
        Mb.n_category;


    Medb = ...
        Mb.category_axis_alignment;


    Yb = ...
        Mb.category_CCGP;


    Covb = [ ...
        Mb.n_identity, ...
        Mb.mean_category_axis_strength ...
        ];



    tbl_ab = table( ...
        Xb, ...
        Covb(:,1), ...
        Covb(:,2), ...
        Medb, ...
        'VariableNames', ...
        { ...
        'n_category', ...
        'n_identity', ...
        'axis_strength', ...
        'alignment' ...
        });



    mdl_ab = fitlm( ...
        tbl_ab, ...
        'alignment ~ n_category + n_identity + axis_strength');



    a_b = mdl_ab.Coefficients.Estimate( ...
        strcmp( ...
        mdl_ab.CoefficientNames, ...
        'n_category'));



    tbl_bb = table( ...
        Xb, ...
        Covb(:,1), ...
        Covb(:,2), ...
        Medb, ...
        Yb, ...
        'VariableNames', ...
        { ...
        'n_category', ...
        'n_identity', ...
        'axis_strength', ...
        'alignment', ...
        'CCGP' ...
        });



    mdl_bb = fitlm( ...
        tbl_bb, ...
        ['CCGP ~ alignment + n_category + ' ...
         'n_identity + axis_strength']);



    b_b = mdl_bb.Coefficients.Estimate( ...
        strcmp( ...
        mdl_bb.CoefficientNames, ...
        'alignment'));



    boot_ab(ib) = ...
        a_b*b_b;

end



CI = ...
    prctile( ...
    boot_ab, ...
    [2.5 97.5]);



fprintf('\nBootstrap indirect effect:\n');


fprintf( ...
    '95%% CI = [%.5f, %.5f]\n', ...
    CI(1), ...
    CI(2));



if CI(1)>0 || CI(2)<0

    fprintf( ...
        'Indirect effect significant: CI excludes zero.\n');

else

    fprintf( ...
        'Indirect effect non-significant: CI includes zero.\n');

end



figure;


histogram( ...
    boot_ab, ...
    40);


xline(0,'--','Zero');


xline( ...
    CI(1), ...
    '--', ...
    '2.5%');


xline( ...
    CI(2), ...
    '--', ...
    '97.5%');


xlabel( ...
    'Indirect effect: category number -> alignment -> CCGP');


ylabel( ...
    'Bootstrap count');


title( ...
    'Held-out-trial mediation analysis');


grid on;










%% ============================================================
% 20. LEAVE-ONE-CATEGORY-NEURON-OUT SENSITIVITY ANALYSIS
%
% PURPOSE
% ------------------------------------------------------------
% Test whether the main differential relationship
%
%       category-axis alignment  >>  category-axis strength
%
% with Category CCGP depends disproportionately on any single
% category-like neuron.
%
% For each jackknife run:
%   1. Remove ONE category-like neuron from the available pool.
%   2. Re-run the complete alignment/strength/CCGP
%      pseudopopulation analysis.
%   3. Use four neuronal compositions:
%
%           CAT0  / ID13
%           CAT5  / ID13
%           CAT10 / ID13
%           CAT12 / ID13
%
%      CAT12 is used instead of CAT13 because one of the
%      original 13 category-like neurons has been removed.
%
%   4. Compute the same zero-order and partial correlations
%      used in the primary analysis.
%
% IMPORTANT
% ------------------------------------------------------------
% This is an influence / robustness analysis.
% It tests whether the alignment > strength pattern depends
% on any single category-like neuron.
%
% It does NOT replace the primary partial-correlation analysis
% and is NOT treated as a new set of independent biological
% observations.
% ============================================================


fprintf('\n====================================================\n');
fprintf('LEAVE-ONE-CATEGORY-NEURON-OUT SENSITIVITY ANALYSIS\n');
fprintf('====================================================\n');


% 20.1 Basic settings

rng(5001);


all_category_neurons = category_neurons(:);

n_category_total = numel(all_category_neurons);


fprintf('\nTotal category-like neurons available: %d\n', ...
    n_category_total);


if n_category_total ~= 13

    warning( ...
        ['Expected 13 category-like neurons based on the primary ' ...
         'analysis, but found %d. The code will continue using ' ...
         'the available number.'], ...
         n_category_total);

end


% Number of pseudopopulation realizations per composition.
% Keep this identical to the primary analysis.
n_jack_iterations = n_iterations;


% Held-out trial count.
K_trials_jack = K_trials;



%% ------------------------------------------------------------
% 20.2 Define jackknife compositions
%
% Identity-like neuron number remains fixed at 13.
%
% Maximum category-like number becomes:
%       total category neurons - 1
%
% With the current dataset:
%       13 - 1 = 12
% ------------------------------------------------------------

max_cat_jack = n_category_total - 1;


if max_cat_jack < 10

    error( ...
        ['After removing one category-like neuron, fewer than ' ...
         '10 category-like neurons remain. The predefined ' ...
         'CAT0/CAT5/CAT10 design cannot be used.']);

end


jack_cat_levels = unique([ ...
    0; ...
    5; ...
    10; ...
    max_cat_jack ...
    ]);


n_jack_scenarios = numel(jack_cat_levels);


fprintf('\nJackknife category compositions:\n');
disp(jack_cat_levels);



% 20.3 Storage tables

jackknife_iteration_results = table();

jackknife_summary = table();



% 20.4 OUTER LOOP:
%      Remove each category-like neuron once

for j = 1:n_category_total


    removed_neuron = ...
        all_category_neurons(j);


    fprintf('\n====================================================\n');

    fprintf( ...
        'Jackknife %d/%d: removing category neuron %d\n', ...
        j, ...
        n_category_total, ...
        removed_neuron);

    fprintf('====================================================\n');


    % Category pool after removing exactly one neuron
    category_pool_jack = ...
        all_category_neurons( ...
        all_category_neurons ~= removed_neuron);


    if numel(category_pool_jack) ~= ...
            (n_category_total - 1)

        error( ...
            'Unexpected category pool size after removing neuron %d.', ...
            removed_neuron);

    end



    % Storage for this leave-one-out dataset

    J = table();



    % Loop across neuronal compositions

    for sc = 1:n_jack_scenarios


        n_cat_this = ...
            jack_cat_levels(sc);


        n_id_this = 13;


        fprintf( ...
            '  Composition: ID%d CAT%d\n', ...
            n_id_this, ...
            n_cat_this);



        % 100 pseudopopulation realizations

        for it = 1:n_jack_iterations


            % Identity-like neurons

            selected_identity = ...
                sample_neurons( ...
                identity_neurons, ...
                n_id_this);



            % Category-like neurons Sample ONLY from the leave-one-out category pool.

            selected_category = ...
                sample_neurons( ...
                category_pool_jack, ...
                n_cat_this);



            % Other neuronal classes: identical to primary analysis

            selected_context = ...
                sample_neurons( ...
                context_neurons, ...
                56);


            selected_response = ...
                sample_neurons( ...
                response_neurons, ...
                13);


            selected_reward = ...
                sample_neurons( ...
                reward_neurons, ...
                13);


            selected_noise = ...
                sample_neurons( ...
                noise_neurons, ...
                100);



            % Build neuronal population

            selected_neurons = unique([ ...
                selected_identity
                selected_category
                selected_context
                selected_response
                selected_reward
                selected_noise ...
                ]);



            % -----------------------------------------------
            % Build pseudopopulation
            %
            % ONLY held-out analysis trials are used,
            % exactly as in the primary analysis.
            % -----------------------------------------------

            [X,labels] = ...
                build_pseudopopulation_independent( ...
                neu, ...
                selected_neurons, ...
                K_trials_jack, ...
                analysis_trials);



            if isempty(X)

                fprintf( ...
                    ['  Skipped jackknife %d, CAT%d, ' ...
                     'iteration %d: insufficient trials.\n'], ...
                    j, ...
                    n_cat_this, ...
                    it);

                continue

            end



            % Category CCGP

            cat_ccgp = ...
                compute_category_CCGP( ...
                X,labels);



            % Category-axis geometry

            geom = ...
                compute_category_axis_alignment( ...
                X,labels);



            % Store realization

            row = table();


            row.jackknife_id = j;

            row.removed_category_neuron = ...
                removed_neuron;


            row.iteration = it;


            row.n_identity = ...
                numel(selected_identity);


            row.n_category = ...
                numel(selected_category);


            row.category_CCGP = ...
                cat_ccgp;


            row.category_axis_alignment = ...
                geom.category_axis_alignment;


            row.mean_category_axis_strength = ...
                geom.mean_category_axis_strength;



            J = [J; row];

        end

    end



    % 20.5 Check this jackknife dataset

    if isempty(J)

        warning( ...
            'No valid realizations for jackknife run %d.', ...
            j);

        continue

    end


    fprintf( ...
        '\nValid realizations in jackknife %d: %d\n', ...
        j, ...
        height(J));



    % 20.6 Zero-order correlations

    [r_align_zero_j, p_align_zero_j] = ...
        corr( ...
        J.category_axis_alignment, ...
        J.category_CCGP, ...
        'Type','Pearson', ...
        'Rows','complete');


    [r_strength_zero_j, p_strength_zero_j] = ...
        corr( ...
        J.mean_category_axis_strength, ...
        J.category_CCGP, ...
        'Type','Pearson', ...
        'Rows','complete');



    %% --------------------------------------------------------
    % 20.7 Partial correlation:
    %
    % Alignment <-> CCGP
    %
    % controlling:
    %   strength
    %   n_category
    %
    % NOTE:
    % n_identity is NOT included because it is constant (=13)
    % across all compositions in this sensitivity analysis.
    % A constant variable provides no information a
    % ---------------------------------------------------------

    [r_align_partial_j, p_align_partial_j] = ...
        partialcorr( ...
        J.category_axis_alignment, ...
        J.category_CCGP, ...
        [ ...
        J.mean_category_axis_strength, ...
        J.n_category ...
        ], ...
        'rows','complete');



    %% --------------------------------------------------------
    % 20.8 Partial correlation:
    %
    % Strength <-> CCGP
    %
    % controlling:
    %   alignment
    %   n_category
    % ---------------------------------------------------------

    [r_strength_partial_j, p_strength_partial_j] = ...
        partialcorr( ...
        J.mean_category_axis_strength, ...
        J.category_CCGP, ...
        [ ...
        J.category_axis_alignment, ...
        J.n_category ...
        ], ...
        'rows','complete');



    %% --------------------------------------------------------
    % 20.9 Difference between partial correlations
    %
    % Positive value means:
    %
    %       alignment association > strength association
    % ---------------------------------------------------------

    delta_partial_r = ...
        r_align_partial_j - ...
        r_strength_partial_j;



    % 20.10 Store jackknife-level summary

    summary_row = table();


    summary_row.jackknife_id = ...
        j;


    summary_row.removed_category_neuron = ...
        removed_neuron;


    summary_row.n_valid_realizations = ...
        height(J);


    summary_row.r_alignment_zero = ...
        r_align_zero_j;


    summary_row.p_alignment_zero = ...
        p_align_zero_j;


    summary_row.r_strength_zero = ...
        r_strength_zero_j;


    summary_row.p_strength_zero = ...
        p_strength_zero_j;


    summary_row.r_alignment_partial = ...
        r_align_partial_j;


    summary_row.p_alignment_partial = ...
        p_align_partial_j;


    summary_row.r_strength_partial = ...
        r_strength_partial_j;


    summary_row.p_strength_partial = ...
        p_strength_partial_j;


    summary_row.delta_partial_r = ...
        delta_partial_r;



    jackknife_summary = ...
        [jackknife_summary; summary_row];


    % Store all realization-level results as well
    jackknife_iteration_results = ...
        [jackknife_iteration_results; J];



    
    % Print current jackknife result

    fprintf('\nJackknife %d result:\n',j);


    fprintf( ...
        '  Removed category neuron = %d\n', ...
        removed_neuron);


    fprintf( ...
        '  Alignment zero-order r = %.3f\n', ...
        r_align_zero_j);


    fprintf( ...
        '  Strength zero-order r = %.3f\n', ...
        r_strength_zero_j);


    fprintf( ...
        '  Alignment partial r = %.3f\n', ...
        r_align_partial_j);


    fprintf( ...
        '  Strength partial r = %.3f\n', ...
        r_strength_partial_j);


    fprintf( ...
        '  Delta partial r = %.3f\n', ...
        delta_partial_r);

end



% 20.11 FINAL JACKKNIFE SUMMARY

fprintf('\n====================================================\n');
fprintf('FINAL LEAVE-ONE-CATEGORY-NEURON-OUT SUMMARY\n');
fprintf('====================================================\n');


disp(jackknife_summary);



% Descriptive summary of partial correlations

mean_align_partial_jack = ...
    mean( ...
    jackknife_summary.r_alignment_partial, ...
    'omitnan');


median_align_partial_jack = ...
    median( ...
    jackknife_summary.r_alignment_partial, ...
    'omitnan');


min_align_partial_jack = ...
    min( ...
    jackknife_summary.r_alignment_partial);


max_align_partial_jack = ...
    max( ...
    jackknife_summary.r_alignment_partial);



mean_strength_partial_jack = ...
    mean( ...
    jackknife_summary.r_strength_partial, ...
    'omitnan');


median_strength_partial_jack = ...
    median( ...
    jackknife_summary.r_strength_partial, ...
    'omitnan');


min_strength_partial_jack = ...
    min( ...
    jackknife_summary.r_strength_partial);


max_strength_partial_jack = ...
    max( ...
    jackknife_summary.r_strength_partial);



mean_delta_jack = ...
    mean( ...
    jackknife_summary.delta_partial_r, ...
    'omitnan');


median_delta_jack = ...
    median( ...
    jackknife_summary.delta_partial_r, ...
    'omitnan');


min_delta_jack = ...
    min( ...
    jackknife_summary.delta_partial_r);


max_delta_jack = ...
    max( ...
    jackknife_summary.delta_partial_r);



fprintf('\nALIGNMENT PARTIAL r ACROSS JACKKNIFE RUNS\n');

fprintf( ...
    'Mean   = %.3f\n', ...
    mean_align_partial_jack);

fprintf( ...
    'Median = %.3f\n', ...
    median_align_partial_jack);

fprintf( ...
    'Range  = [%.3f, %.3f]\n', ...
    min_align_partial_jack, ...
    max_align_partial_jack);



fprintf('\nSTRENGTH PARTIAL r ACROSS JACKKNIFE RUNS\n');

fprintf( ...
    'Mean   = %.3f\n', ...
    mean_strength_partial_jack);

fprintf( ...
    'Median = %.3f\n', ...
    median_strength_partial_jack);

fprintf( ...
    'Range  = [%.3f, %.3f]\n', ...
    min_strength_partial_jack, ...
    max_strength_partial_jack);



fprintf('\nDIFFERENCE: ALIGNMENT r - STRENGTH r\n');

fprintf( ...
    'Mean   = %.3f\n', ...
    mean_delta_jack);

fprintf( ...
    'Median = %.3f\n', ...
    median_delta_jack);

fprintf( ...
    'Range  = [%.3f, %.3f]\n', ...
    min_delta_jack, ...
    max_delta_jack);




% 20.12 How consistently is alignment > strength?

valid_delta = ...
    jackknife_summary.delta_partial_r( ...
    ~isnan(jackknife_summary.delta_partial_r));


n_alignment_greater = ...
    sum(valid_delta > 0);


prop_alignment_greater = ...
    n_alignment_greater / ...
    numel(valid_delta);


fprintf('\n====================================================\n');

fprintf( ...
    ['Alignment partial r > strength partial r in ' ...
     '%d/%d jackknife runs (%.1f%%).\n'], ...
    n_alignment_greater, ...
    numel(valid_delta), ...
    100*prop_alignment_greater);

fprintf('====================================================\n');



% 20.13 JACKKNIFE FIGURE

figure;


jack_x = ...
    1:height(jackknife_summary);


plot( ...
    jack_x, ...
    jackknife_summary.r_alignment_partial, ...
    '-o', ...
    'LineWidth',1.5);


hold on;


plot( ...
    jack_x, ...
    jackknife_summary.r_strength_partial, ...
    '-s', ...
    'LineWidth',1.5);


yline( ...
    0.731, ...
    '--', ...
    'Full-data alignment');

yline( ...
    0.110, ...
    '--', ...
    'Full-data strength');

xlabel('Removed category-like neuron');


ylabel('Partial correlation with CCGP');


xticks(jack_x);


xticklabels( ...
    string( ...
    jackknife_summary.removed_category_neuron));


xtickangle(45);


legend( ...
    'Alignment', ...
    'Strength', ...
    'Full-data alignment', ...
    'Full-data strength', ...
    'Location','best');


title( ...
    'Leave-one-category-neuron-out sensitivity analysis');


grid on;



% 20. DELTA FIGURE

figure;


plot( ...
    jack_x, ...
    jackknife_summary.delta_partial_r, ...
    '-o', ...
    'LineWidth',1.5);


hold on;


yline( ...
    0, ...
    '--', ...
    'Equal association');


xlabel('Removed category-like neuron');


ylabel( ...
    '\Delta partial r: alignment - strength');


xticks(jack_x);


xticklabels( ...
    string( ...
    jackknife_summary.removed_category_neuron));


xtickangle(45);


title( ...
    'Robustness of alignment-strength difference');


grid on;