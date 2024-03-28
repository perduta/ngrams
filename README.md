# Simplified N-gram Language Model

This project implements a simplified N-gram language model for text generation and perplexity calculation. The model is trained on the WikiText-2 dataset and can generate text based on a given prefix and calculate the perplexity of the model on the test dataset.

## Implementation Details

The implemented solution follows these steps:

1. Preprocess the text data by tokenizing, converting to lowercase, removing punctuation and special characters, and removing stopwords.
2. Generate N-grams from the preprocessed text tokens.
3. Count the occurrences of each N-gram.
4. Calculate the probabilities of each word given its context (previous N-1 words).
5. Build the N-gram language model using the calculated probabilities.
6. Generate text based on a given prefix and the trained model.
7. Calculate the perplexity of the model on the test dataset.

## Instructions for Running the Program

```bash
python3.12 -m venv && source venv/bin/activate # or alternative, code hasn't been tested on lower versions of python
pip install datasets nltk
python main.py
```

## Example Generated Texts

Here are some example generated texts for different values of n:

### For n = 2:
Prefix: "valkyria 3 unk chronicles japanese"
Generated text: "valkyria 3 unk chronicles japanese unk unk unk unk unk unk unk unk unk unk"

### For n = 3:
Prefix: "valkyria 3 unk chronicles japanese"
Generated text: "valkyria 3 unk chronicles japanese 戦場のヴァルキュリア3 lit valkyria battlefield 3 unk w 51 unk n"

### For n = 4:
Prefix: "valkyria 3 unk chronicles japanese"
Generated text: "valkyria 3 unk chronicles japanese 戦場のヴァルキュリア3 lit valkyria battlefield 3 unk taken unk sake originally"

## Perplexity Results

The perplexity results on the train and test datasets for different values of n are as follows:

- For n = 2: Perplexity (train): 74.34690101727507, Perplexity (test): 2290940.4566380545
- For n = 3: Perplexity (train): 1.8864595101190142, Perplexity (test): 760460392.8112597
- For n = 4: Perplexity (train): 1.081815014484215, Perplexity (test): 1665697646.7602937

## Conclusions and Observations

Based on the generated texts and perplexity results, we can make the following observations:

1. As the value of n increases, the model tends to overfit to the training data. This is evident from the significantly lower perplexity values on the training dataset compared to the rapidly increasing perplexity values on the test dataset.
2. The n-gram model with n = 2 appears to have the best performance on the test dataset, despite generating nonsensical predictions. This suggests that the model with n = 2 is able to capture the general word distribution of the test set, even though it lacks the ability to generate coherent and meaningful text.
3. The rapid increase in perplexity values on the test dataset for higher values of n indicates that the model becomes overly specific to the training data and struggles to generalize well to unseen contexts. This overfitting behavior can be attributed to the limited size of the dataset (WikiText-2 contains approximately 1 million tokens).

It can be argued that if the dataset size were larger, the optimal value of n for minimizing perplexity on the test set would likely be higher. With more training data, the model would have a better chance of capturing more diverse and representative N-grams, allowing it to generalize better to unseen data.

In conclusion, the experiment highlights the trade-off between model complexity and generalization ability when using N-gram language models. While higher values of n can capture more context and generate more coherent text, they also tend to overfit to the training data when the dataset is limited in size. The optimal choice of n depends on the specific dataset and the desired balance between model performance and generalization capability. In this case, the limited size of the WikiText-2 dataset favors lower values of n, with n = 2 achieving the lowest perplexity on the test set despite generating less meaningful text.