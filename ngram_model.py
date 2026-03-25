import nltk
import re
import math
import random
from collections import defaultdict, Counter

"""# Load Corpus"""

nltk.download('gutenberg')
from nltk.corpus import gutenberg

V = []

"""# Preprocessing"""

def preprocess(text: str) -> list[str]:
    """Returns lowercase word tokens with no punctuation."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # tokens = text.split()
    tokens = nltk.word_tokenize(text)
    return tokens

"""# Build Trigram Model"""

def build_trigram_model(tokens: list[str]) -> dict:
    """Returns trigram_counts[(w1,w2)][next_word] = count."""
    trigram_counts = defaultdict(lambda: defaultdict(int))

    for i in range(len(tokens) - 2):
        w1 = tokens[i]
        w2 = tokens[i + 1]
        w3 = tokens[i + 2]
        trigram_counts[(w1, w2)][w3] += 1

    return trigram_counts

"""# Laplace Smoothing"""


def laplace_smoothing(trigram_counts: dict, vocab_size: int) -> dict:
    """Returns smoothed_probs[(w1,w2)][next_word] = probability."""
    smoothed_probs = {}

    vocab_list = V if V else sorted(
        {
            word
            for context in trigram_counts
            for word in trigram_counts[context]
        }
    )


    for context in trigram_counts:
        smoothed_probs[context] = {}
        context_total = sum(trigram_counts[context].values())

        for word in vocab_list:
            count = trigram_counts[context].get(word, 0)
            smoothed_probs[context][word] = (count + 1) / (context_total + vocab_size)

    return smoothed_probs

"""# Text Generation"""


def generate_text(seed: list[str], smoothed_probs: dict,
                  vocab: list[str], num_words: int = 30) -> str:
    """Returns a generated story string of num_words words."""
    generated = seed.copy()

    while len(generated) < num_words:
        context = (generated[-2], generated[-1])
        if context in smoothed_probs and smoothed_probs[context]:
            next_word = max(smoothed_probs[context], key=smoothed_probs[context].get)
        else:
            next_word = random.choice(vocab)

        generated.append(next_word)

    return " ".join(generated)

"""# Perplexity Evaluation"""

def compute_perplexity(test_tokens: list[str], smoothed_probs: dict,
                       vocab_size: int) -> float:
    """Returns perplexity as a float."""
    if len(test_tokens) < 3:
        return float("inf")

    log_prob_sum = 0.0
    prediction_count = len(test_tokens) - 2

    for i in range(2, len(test_tokens)):
        context = (test_tokens[i - 2], test_tokens[i - 1])
        word = test_tokens[i]

        if context in smoothed_probs:
            prob = smoothed_probs[context].get(word, 1 / vocab_size)
        else:
            prob = 1 / vocab_size

        prob = max(prob, 1e-10)
        # print(prob)
        log_prob_sum += math.log(prob)

    perplexity = math.exp(-log_prob_sum / prediction_count)
    return perplexity

"""# Bonus - Simple Linear Interpolation"""

def build_unigram_model(tokens):
    """Returns unigram_counts[word] = count."""
    return Counter(tokens)



def build_bigram_model(tokens):
    """Returns bigram_counts[w1][w2] = count."""
    bigram_counts = defaultdict(lambda: defaultdict(int))

    for i in range(len(tokens) - 1):
        w1 = tokens[i]
        w2 = tokens[i + 1]
        bigram_counts[w1][w2] += 1

    return bigram_counts

def interpolated_probability(w, w1, w2,
                             trigram_counts,
                             bigram_counts,
                             unigram_counts,
                             vocab_size,
                             total_tokens,
                             trigram_weight=0.6,
                             bigram_weight=0.3,
                             unigram_weight=0.1):
    """Returns interpolated probability using trigram, bigram, unigram models."""

    tri_total = sum(trigram_counts.get((w1, w2), {}).values())
    tri_count = trigram_counts.get((w1, w2), {}).get(w, 0)
    p_tri = tri_count / tri_total if tri_total > 0 else 0.0

    bi_total = sum(bigram_counts.get(w2, {}).values())
    bi_count = bigram_counts.get(w2, {}).get(w, 0)
    p_bi = bi_count / bi_total if bi_total > 0 else 0.0

    p_uni = unigram_counts.get(w, 0) / total_tokens if total_tokens > 0 else 0.0

    return (
        trigram_weight * p_tri
        + bigram_weight * p_bi
        + unigram_weight * p_uni
    )


def generate_text_interpolation(seed, tokens, trigram_counts,
                                bigram_counts, unigram_counts,
                                vocab, num_words=30):
    """Generates text using interpolated probabilities."""
    generated = seed.copy()
    total_tokens = len(tokens)

    while len(generated) < num_words:
        w1, w2 = generated[-2], generated[-1]
        probs = {}

        for word in vocab:
            probs[word] = interpolated_probability(
                word, w1, w2,
                trigram_counts,
                bigram_counts,
                unigram_counts,
                len(vocab),
                total_tokens
            )

        next_word = max(probs, key=probs.get)
        generated.append(next_word)

    return " ".join(generated)

def compute_perplexity_interpolation(test_tokens, tokens,
                                     trigram_counts,
                                     bigram_counts,
                                     unigram_counts,
                                     vocab):
    """Computes perplexity using interpolated probabilities."""
    if len(test_tokens) < 3:
        return float("inf")

    total_tokens = len(tokens)
    log_prob_sum = 0.0
    prediction_count = len(test_tokens) - 2

    for i in range(2, len(test_tokens)):
        w1 = test_tokens[i - 2]
        w2 = test_tokens[i - 1]
        w = test_tokens[i]

        prob = interpolated_probability(
            w, w1, w2,
            trigram_counts,
            bigram_counts,
            unigram_counts,
            len(vocab),
            total_tokens
        )

        prob = max(prob, 1e-10)
        log_prob_sum += math.log(prob)

    return math.exp(-log_prob_sum / prediction_count)

"""# Main"""

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('punkt_tab')
    random.seed(42)

    raw_text = gutenberg.raw('shakespeare-caesar.txt')
    tokens = preprocess(raw_text)

    V = sorted(set(tokens))
    trigram_counts = build_trigram_model(tokens)
    smoothed_probs = laplace_smoothing(trigram_counts, len(V))

    # preferred_seed = ['dark', 'king']
    # seed = preferred_seed
    # if not seed:
    #     seed = [random.choice(V), random.choice(V)]
    # elif len(seed) == 1:
    #     seed = [seed[0], random.choice(V)]
    # else:
    #     seed = list(seed[:2])

    preferred_seed = ['i', 'will']
    if tuple(preferred_seed) in trigram_counts:
      seed = preferred_seed
    else:
      seed = list(random.choice(list(trigram_counts.keys())))

    generated_laplace = generate_text(seed, smoothed_probs, V)

    test_sentence = "the king is dead"
    test_tokens = preprocess(test_sentence)
    perplexity_laplace = compute_perplexity(
        test_tokens, smoothed_probs, len(V)
    )

    unigram_counts = build_unigram_model(tokens)
    bigram_counts = build_bigram_model(tokens)
    generated_interp = generate_text_interpolation(
        seed, tokens,
        trigram_counts,
        bigram_counts,
        unigram_counts,
        V
    )

    perplexity_interp = compute_perplexity_interpolation(
        test_tokens, tokens,
        trigram_counts,
        bigram_counts,
        unigram_counts,
        V
    )
    print("=== Preprocessing ===")
    print(f"Vocabulary size: {len(V)}")
    print(f"Total tokens:    {len(tokens)}")

    print("\n=== Text Generation (Laplace Smoothing) ===")
    print(f"Seed: {' '.join(seed)}")
    print(f"Generated: {generated_laplace}")

    print("\n=== Perplexity ===")
    print("Test sentence: 'the king is dead'")
    print(f"Perplexity: {perplexity_laplace}")

    print("\n=== Bonus (if attempted) ===")
    print(f"Generated (Interpolation): {generated_interp}")
    print("Sentence: 'the king is dead'")
    print(f"Perplexity (Laplace): {perplexity_laplace}")
    print(f"Perplexity (Interpolation): {perplexity_interp}")

    print("\nComparison:")
    print("Laplace smoothing assigns probability mass to unseen trigrams, so it is safer but often more generic.")
    print("Simple linear interpolation uses trigram, bigram, and unigram probabilities together, which usually makes the model more flexible.")
    print("Because interpolation uses multiple context levels, it often gives a more stable perplexity score on short test sentences.")
    print("In generation, interpolation can produce text that feels more context-aware than Laplace-only generation.")