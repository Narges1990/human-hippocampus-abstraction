function c = cosine_similarity(a,b)

    c = dot(a,b) / ((norm(a) * norm(b)) + eps);

end