function geom = compute_category_axis_alignment(X, labels)

    % z-score neurons
    Xz = zscore(X, 0, 1);

    % Category axis in context 1
    idx_c1_cat1 = labels.context==1 & labels.category==1;
    idx_c1_cat2 = labels.context==1 & labels.category==2;

    v_cat_ctx1 = mean(Xz(idx_c1_cat1,:),1) - mean(Xz(idx_c1_cat2,:),1);

    % Category axis in context 2
    idx_c2_cat1 = labels.context==2 & labels.category==1;
    idx_c2_cat2 = labels.context==2 & labels.category==2;

    v_cat_ctx2 = mean(Xz(idx_c2_cat1,:),1) - mean(Xz(idx_c2_cat2,:),1);

    % Alignment between category axes across contexts
    geom.category_axis_alignment = cosine_similarity(v_cat_ctx1, v_cat_ctx2);

    geom.category_axis_strength_ctx1 = norm(v_cat_ctx1);
    geom.category_axis_strength_ctx2 = norm(v_cat_ctx2);
    geom.mean_category_axis_strength = mean([ ...
        geom.category_axis_strength_ctx1, ...
        geom.category_axis_strength_ctx2]);

    % Context axis for category 1
    idx_cat1_ctx1 = labels.category==1 & labels.context==1;
    idx_cat1_ctx2 = labels.category==1 & labels.context==2;

    v_ctx_cat1 = mean(Xz(idx_cat1_ctx1,:),1) - mean(Xz(idx_cat1_ctx2,:),1);

    % Context axis for category 2
    idx_cat2_ctx1 = labels.category==2 & labels.context==1;
    idx_cat2_ctx2 = labels.category==2 & labels.context==2;

    v_ctx_cat2 = mean(Xz(idx_cat2_ctx1,:),1) - mean(Xz(idx_cat2_ctx2,:),1);

    geom.context_axis_alignment = cosine_similarity(v_ctx_cat1, v_ctx_cat2);

    geom.context_axis_strength_stim13 = norm(v_ctx_cat1);
    geom.context_axis_strength_stim24 = norm(v_ctx_cat2);
    geom.mean_context_axis_strength = mean([ ...
        geom.context_axis_strength_stim13, ...
        geom.context_axis_strength_stim24]);

end




