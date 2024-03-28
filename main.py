from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datasets import load_dataset
from collections import defaultdict
import math
import nltk

def preprocess_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text)
    
    # Convert all tokens to lowercase
    tokens = [token.lower() for token in tokens]
    
    # Remove punctuation and special characters
    tokens = [token for token in tokens if token.isalnum()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return tokens

def generate_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

def count_ngrams(ngrams):
    counts = defaultdict(int)
    for ngram in ngrams:
        counts[ngram] += 1
    return counts

def calculate_probabilities(ngram_counts, n):
    probabilities = defaultdict(dict)
    prefix_counts = defaultdict(int)

    # Calculate prefix counts
    for ngram, count in ngram_counts.items():
        prefix = ngram[:-1]
        prefix_counts[prefix] += count

    # Calculate probabilities
    for ngram, count in ngram_counts.items():
        prefix = ngram[:-1]
        next_word = ngram[-1]
        probabilities[prefix][next_word] = count / prefix_counts[prefix]

    return probabilities

def build_ngram_model(text, n):
    tokens = preprocess_text(text)
    ngrams = generate_ngrams(tokens, n)
    ngram_counts = count_ngrams(ngrams)
    probabilities = calculate_probabilities(ngram_counts, n)
    return probabilities

def generate_text(model, prefix, max_words, n):
    generated_text = list(prefix)
    for _ in range(max_words):
        context = tuple(generated_text[-(n-1):])
        if context in model:
            preds = sorted(model[context].items(), key=lambda x: x[1], reverse=True)[:5]
            next_word = preds[0][0]
            generated_text.append(next_word)
        else:
            break
    return ' '.join(generated_text)

def calculate_perplexity(model, test_data, n):
    log_likelihood = 0
    num_words = 0
    for tokens in test_data:
        num_words += len(tokens)
        for i in range(n - 1, len(tokens)):
            context = tuple(tokens[i - n + 1:i])
            word = tokens[i]
            if context in model:
                if word in model[context]:
                    probability = model[context][word]
                    log_likelihood += math.log(probability)
                else:
                    # Handle unseen words
                    log_likelihood += math.log(1e-10)  # Assign a small probability to avoid zero
            else:
                # Handle unseen contexts
                log_likelihood += math.log(1e-10)  # Assign a small probability to avoid zero
    perplexity = math.exp(-log_likelihood / num_words)
    return perplexity

def run_experiment(n):
    print(f"Running experiment for n-gram {n}")
    text = ' '.join([token for sample in dataset["train"]['text'] for token in sample])
    model = build_ngram_model(text, n)

    # prefix = input("Enter the beginning of a sentence: ").split()
    prefix = "valkyria 3 unk chronicles japanese".split()
    print("Prefix: ", prefix)
    max_words = 10  # Change this to generate longer or shorter text
    generated_text = generate_text(model, prefix, max_words, n)
    print("Generated text:", generated_text)

    # Perplexity calculation
    test_data = dataset["test"]["text"]
    perplexity_train = calculate_perplexity(model, dataset["train"]["text"], n)
    perplexity = calculate_perplexity(model, test_data, n)
    print(f"Perplexity (train): {perplexity_train}")
    print(f"Perplexity (test): {perplexity}")

    return model


if __name__ == "__main__":
    dataset = load_dataset("wikitext", 'wikitext-2-v1')
    nltk.download('punkt')
    nltk.download('stopwords')
    dataset = dataset.map(lambda example: {'text': preprocess_text(example['text'])})

    # for n in [2, 3, 4]:
    #     run_experiment(n)

    n = 2
    model = run_experiment(n)
    while True:
        prefix = input("Enter the beginning of a sentence: ").split()
        max_words = 10  # Change this to generate longer or shorter text
        generated_text = generate_text(model, prefix, max_words, n)
        print("Generated text:", generated_text)