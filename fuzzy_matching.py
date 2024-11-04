import re
from collections import Counter
import math
import numpy as np

def damerau_levenshtein(s1, s2):            # This function uses dynamic programming.

    d = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]           # Creating a matrix of size (len(s1)+1) x (len(s2)+1).

    for i in range(len(s1) + 1):            # Initialization of the first column and the first row.
        d[i][0] = i
    for j in range(len(s2) + 1):
        d[0][j] = j

    for i in range(1, len(s1) + 1):         # Fill the matrix.
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1

            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)         # Calculating the minimum value between deletion (first), insertion (second), and replacement (third) of a character.

            if i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and s1[i - 2] == s2[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)  # Transposition.

    return d[len(s1)][len(s2)]

def scaled_damerau_levenshtein(s1, s2):     # This function scales the Damerau-Levenshtein distance to the range [0, 1]. It ensures that similar strings yield values closer to 1, while dissimilar strings yield values closer to 0.

    max_length = max(len(s1), len(s2))
    if max_length == 0:
        return 1
    distance = damerau_levenshtein(s1, s2)
    return 1 - (distance / max_length)

def jaro_winkler(s1, s2, p=0.1, max_prefix=4):

    len_s1 = len(s1)
    len_s2 = len(s2)

    match_distance = max(len_s1, len_s2) // 2 - 1           # Maximum distance for matching characters.

    s1_matches = [False] * len_s1           # Initializing lists to track matching characters.
    s2_matches = [False] * len_s2
    matches = 0

    for i in range(len_s1):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_s2)
        for j in range(start, end):
            if s2_matches[j]:
                continue
            if s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    t = 0
    s2_match_index = 0
    for i in range(len_s1):             # Calculating the number of transpositions.
        if s1_matches[i]:
            while not s2_matches[s2_match_index]:
                s2_match_index += 1
            if s1[i] != s2[s2_match_index]:
                t += 1
            s2_match_index += 1

    t = t // 2  # Half of transpositions.

    jaro_distance = (matches / len_s1 + matches / len_s2 + (matches - t) / matches) / 3             # Jaro Distance formula.

    prefix_length = 0
    for i in range(min(len_s1, len_s2)):            # Reward for matching initial characters.
        if s1[i] == s2[i]:
            prefix_length += 1
        else:
            break
        if prefix_length == max_prefix:
            break

    jaro_winkler_distance = jaro_distance + (prefix_length * p * (1 - jaro_distance))    # Jaro-Winkler formula.

    return jaro_winkler_distance



interpuctions = [" ", ".", "-", "_", "!", "?", "`", "'", ";", ":"]

def split_by_punctuation(s):

    pattern = "[" + re.escape("".join(interpuctions)) + "]"             # Create a regex pattern that includes all punctuation characters.

    split_result = re.split(pattern, s)             # Split the string using the regular expression.

    return [item for item in split_result if item]          # Remove empty strings from the result.


def create_word_vector(bow, combined_vocab):

    word_count = Counter(bow)           # Count the frequency of each word (bag of words).

    word_vector = [word_count[word] for word in combined_vocab]         # Create a vector containing the occurrence count for each word from the combined_vocabulary.

    return word_vector

def scalar_product(vector1, vector2):           # Returns the scalar product of the two vectors.
    return np.dot(vector1, vector2)

def vector_intensity(vector):           # Calculate the norm of the vector.
    return math.sqrt(sum(x ** 2 for x in vector))

def cosine_similarity(s1, s2):              # Based on bag of words.

    s1_bow = split_by_punctuation(s1)           # Split the input strings into bags of words by removing punctuation.
    s2_bow = split_by_punctuation(s2)

    combined_vocab = list(set(s1_bow).union(set(s2_bow)))          # Create a combined vocabulary of words from both bags of words.

    s1_vector = create_word_vector(s1_bow, combined_vocab)             # Create word vectors for each bag of words based on the combined vocabulary.
    s2_vector = create_word_vector(s2_bow, combined_vocab)

    scalar_value = scalar_product(s1_vector, s2_vector)             # Calculate the scalar product of the two vectors

    s1_intensity = vector_intensity(s1_vector)              # Calculate the intensity (norm) of each vector
    s2_intensity = vector_intensity(s2_vector)

    cos = scalar_value/(s1_intensity*s2_intensity)

    return cos

def calculate_heuristic(dl_score, jw_score, cos_score):             # Calculate a linear combination of similarity scores based on predefined priorities.
    return 0.5 * jw_score + 0.3 * dl_score + 0.2 * cos_score

def calculate_f_measure(predicted, true, beta):             # Beta parameter controls the balance between precision and recall.
    # Calculate true positives, false positives, and false negatives.
    tp = len([p for p in predicted if p in true])  # True positives.
    fp = len([p for p in predicted if p not in true])  # False positives.
    fn = len([t for t in true if t not in predicted])  # False negatives.

    precision = tp / (tp + fp) if tp + fp > 0 else 0            # Calculate precision: the ratio of true positives to all predicted positives.
    recall = tp / (tp + fn) if tp + fn > 0 else 0               # Calculate recall: the ratio of true positives to all actual positives.

    if precision + recall == 0:
        f_beta_score = 0
    else:
        f_beta_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)      # Calculate f beta measure

    return f_beta_score

def calculate_macro_f_measure(true_matches, predictions, beta=1):

    total_f_beta = 0
    num_companies = len(true_matches)

    for company, true_match in true_matches.items():
        predicted = predictions[company]
        f_beta_score = calculate_f_measure(predicted, true_match, beta)
        total_f_beta += f_beta_score

    # Calculating the average (macro) precision as the arithmetic mean of all F measures.
    macro_f1 = total_f_beta / num_companies

    return macro_f1

true_matches = {
    "Coca Cola": ["Coca-Cola", "CocaCola", "Coca Cola Company"],
    "Tesla": ["Tesla Motors", "Tesla, Inc.", "Tesla Inc"],
    "Apple Inc.": ["Apple", "Apple Computer Inc.", "Apple Inc"]
}

if __name__=="__main__":
    list_a = ["Coca Cola", "Tesla", "Apple Inc."]
    list_b = ["Coca-Cola", "CocaCola", "Tesla Motors", "Apple", "Apple Computer Inc.", "Tesla, Inc.", "PepsiCo", "Coca-Cola Company", "Tesla Inc", "Apple Inc", "Coca Cola Company", "CocaCola Company", "Tesla Motors, Inc.", "Apple Computers", "Coca Cola Beverages"]

    predictions = {}

    for company_a in list_a:
        similarity_scores = []
        for company_b in list_b:
            dl_score = scaled_damerau_levenshtein(company_a, company_b)
            jw_score = jaro_winkler(company_a, company_b)
            cos_score = cosine_similarity(company_a, company_b)
            heuristic_score = calculate_heuristic(dl_score, jw_score, cos_score)          # Calculate the overall similarity using heuristics.

            similarity_scores.append((company_b, heuristic_score))

        top_company = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:3]           # Sort and get the top 3 with the highest heuristic value.
        predictions[company_a] = [company for company, _ in top_company]

        print(f"Best matches for {company_a}:")
        for company, score in top_company:
            print(f"  - {company}: {round(score, 2)}")

    macro_f = calculate_macro_f_measure(true_matches, predictions)

    print(f"Macro F measure: {macro_f:.2f}")
