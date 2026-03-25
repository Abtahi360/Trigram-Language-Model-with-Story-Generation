## 📖 Overview

The goal of this project is to understand how **n-gram language models** work by building a trigram-based text generator without using any built-in n-gram libraries.

The model:

* Learns word sequences from a dataset
* Applies smoothing techniques to handle unseen data
* Generates a 30-word story from a given seed
* Evaluates performance using perplexity

---

## 🚀 Features

* ✅ Text preprocessing (lowercasing, punctuation removal, tokenization)
* ✅ Trigram frequency model using dictionaries
* ✅ Laplace (Add-1) smoothing implementation
* ✅ Greedy text generation (highest probability selection)
* ✅ Perplexity calculation for evaluation
* ⭐ Bonus: Simple Linear Interpolation (unigram + bigram + trigram)

---

## 📊 Dataset

This project uses the **NLTK Gutenberg corpus**, specifically:

* *Shakespeare – Julius Caesar*

---

## 🧠 How It Works

### 1. Preprocessing

* Converts text to lowercase
* Removes punctuation
* Tokenizes into words

### 2. Trigram Model

Builds frequency counts for:

```
P(word | previous two words)
```

### 3. Laplace Smoothing

Handles unseen word sequences using:

```
(count + 1) / (context_total + vocabulary_size)
```

### 4. Text Generation

* Starts from a seed (e.g., *"the king"*)
* Generates 30 words
* Uses greedy selection (highest probability)

### 5. Perplexity

Evaluates model performance:

* Lower perplexity = better prediction

---

## ⭐ Bonus: Interpolation

Implements **Simple Linear Interpolation (SLI)** using:

* Unigram
* Bigram
* Trigram

Formula:

```
P = λ₃·P(trigram) + λ₂·P(bigram) + λ₁·P(unigram)
```

---

## 📌 Sample Output

```
=== Preprocessing ===
Vocabulary size: 8000
Total tokens:    30000

=== Text Generation (Laplace Smoothing) ===
Seed: the king
Generated: the king ...

=== Perplexity ===
Test sentence: 'the king is dead'
Perplexity: 120.45
```

---

## 🎯 Learning Outcomes

* Understanding statistical language models
* Handling zero-probability problems
* Applying smoothing techniques
* Evaluating NLP models using perplexity

---

## ⚠️ Notes

* No built-in n-gram functions were used
* All implementations are written from scratch
* Designed for educational purposes
