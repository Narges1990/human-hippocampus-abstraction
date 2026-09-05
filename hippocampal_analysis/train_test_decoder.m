function acc = train_test_decoder(X, y, train_idx, test_idx)

    X_train = X(train_idx,:);
    X_test  = X(test_idx,:);

    y_train = y(train_idx);
    y_test  = y(test_idx);

    [X_train, X_test] = zscore_train_test(X_train, X_test);

    mdl = fitclinear(X_train, y_train, ...
        'Learner','logistic', ...
        'Regularization','ridge');

    y_pred = predict(mdl, X_test);

    acc = mean(y_pred == y_test);

end

