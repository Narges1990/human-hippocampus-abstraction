function acc = compute_context_CCGP(X, labels)

    % Train on stim 1,2; test on stim 3,4
    train_idx = labels.stim == 1 | labels.stim == 2;
    test_idx  = labels.stim == 3 | labels.stim == 4;

    acc1 = train_test_decoder(X, labels.context, train_idx, test_idx);

    % Train on stim 3,4; test on stim 1,2
    train_idx = labels.stim == 3 | labels.stim == 4;
    test_idx  = labels.stim == 1 | labels.stim == 2;

    acc2 = train_test_decoder(X, labels.context, train_idx, test_idx);

    acc = mean([acc1 acc2]);

end
