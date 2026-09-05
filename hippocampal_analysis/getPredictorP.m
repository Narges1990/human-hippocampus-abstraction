function p = getPredictorP(mdl, predictorName)

    coefNames = mdl.CoefficientNames;

    idx = startsWith(coefNames, predictorName);

    % Remove intercept if accidentally selected
    idx(strcmp(coefNames, '(Intercept)')) = false;

    H = zeros(sum(idx), numel(coefNames));

    rows = find(idx);

    for r = 1:numel(rows)
        H(r, rows(r)) = 1;
    end

    if isempty(H)
        p = NaN;
    else
        p = coefTest(mdl, H);
    end

end