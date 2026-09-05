function sampled = sample_neurons(pool, n)

    pool = pool(:);

    if n == 0
        sampled = [];
        return
    end

    n = min(n, numel(pool));
    sampled = pool(randperm(numel(pool), n));

end


