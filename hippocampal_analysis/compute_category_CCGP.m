

function acc = compute_category_CCGP(X, labels)

    % Train on context 1, test on context 2
    train_idx = labels.context == 1;
    test_idx  = labels.context == 2;

    [acc1] = train_test_decoder(X, labels.category, train_idx, test_idx);

    % Train on context 2, test on context 1
    train_idx = labels.context == 2;
    test_idx  = labels.context == 1;

    [acc2] = train_test_decoder(X, labels.category, train_idx, test_idx);

    acc = mean([acc1 acc2]);

end
