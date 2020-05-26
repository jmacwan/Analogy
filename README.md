# Analogy
Analogy is an experimental open source project for Natural Language Processing. It aims to perform 2 newly introduced NLP tasks: word comparison and sentence comparison.

Analogy provides semantic similiarity and differences between two pieces of text. Text can be in the form of a word or a sentence.

A pretrained model is released to get easily started. You can also retrain upon an existing model.

### Getting Started:


#### Prerequisites:
Python 3.0 or higher

Stanford Core NLP (3.9.2)

#### Installing:
```
pip install analogy
```
Read instructions on [how to install and run Stanford CoreNLP server](http://stanfordnlp.github.io/CoreNLP/corenlp-server.html#getting-started) 

#### Analogy functions:

1. findComparison(model, word1, word2)
2. findSentenceComparison(model, sentence1, sentence2)
3. trainModel(sentences) #Input is list of sentences
4. retrainModel(model, sentences)
5. saveModel(name, model) #Be sure to add '.npz' at the end
6. loadModel(name)

#### Example:
```
findComparison(model, "apple", "orange")
```
#### Output:
```
Word1 = apple
Word2 = orange
Similiarity = fruit
```
