
function C = compute_condition_means(X, labels)

    C = zeros(8, size(X,2));

    cnt = 1;

    for ctx = 1:2
        for stim = 1:4

            idx = labels.context == ctx & labels.stim == stim;

            C(cnt,:) = mean(X(idx,:),1);

            cnt = cnt + 1;

        end
    end
end