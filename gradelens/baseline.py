from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def grade_by_reference(answer, reference, max_score=3):
    if not answer.strip() or not reference.strip():
        return 0, 0.0
    try:
        matrix = TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b").fit_transform(
            [answer.lower(), reference.lower()])
        similarity = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
    except ValueError:
        similarity = 0.0
    return min(max_score, max(0, round(similarity * max_score))), similarity
