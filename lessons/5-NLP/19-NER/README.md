# Named Entity Recognition

Up to now, we have mostly been concentrating on one NLP task - classification. However, there are also other NLP tasks that can be accomplished with neural networks. One of those tasks is **[Named Entity Recognition](https://wikipedia.org/wiki/Named-entity_recognition)** (NER), which deals with recognizing specific entities within text, such as places, person names, date-time intervals, chemical formulae and so on.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ai/quiz/37)

## Example of Using NER

Suppose you want to develop a natural language chat bot, similar to Amazon Alexa or Google Assistant. The way intelligent chat bots work is to *understand* what the user wants by doing text classification on the input sentence. The result of this classification is so-called **intent**, which determines what a chat bot should do.

<img alt="Bot NER" src="images/bot-ner.png" width="50%"/>

> Image by the author

However, a user may provide some parameters as part of the phrase. For example, when asking for the weather, she may specify a location or date. A bot should be able to understand those entities, and fill in the parameter slots accordingly before performing the action. This is exactly where NER comes in.

> ✅ Another example would be [analyzing scientific medical papers](https://soshnikov.com/science/analyzing-medical-papers-with-azure-and-text-analytics-for-health/). One of the main things we need to look for are specific medical terms, such as diseases and medical substances. While a small number of diseases can probably be extracted using substring search, more complex entities, such as chemical compounds and medication names, need a more complex approach.

## NER as Token Classification

NER models are essentially **token classification models**, because for each of the input tokens we need to decide whether it belongs to an entity or not, and if it does - to which entity class.

Consider the following paper title:

**Tricuspid valve regurgitation** and **lithium carbonate** **toxicity** in a newborn infant.

Entities here are:

* Tricuspid valve regurgitation is a disease (`DIS`)
* Lithium carbonate is a chemical substance (`CHEM`)
* Toxicity is also a disease (`DIS`)

Notice that one entity can span several tokens. And, as in this case, we need to distinguish between two consecutive entities. Thus, it is common to use two classes for each entity - one specifying the first token of the entity (often the `B-` prefix is used, for **b**eginning), and another - the continuation of an entity (`I-`, for **i**nner token). We also use `O` as a class to represent all **o**ther tokens. Such token tagging is called [BIO tagging](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) (or IOB). When tagged, our title will look like this:

Token | Tag
------|-----
Tricuspid | B-DIS
valve | I-DIS
regurgitation | I-DIS
and | O
lithium | B-CHEM
carbonate | I-CHEM
toxicity | B-DIS
in | O
a | O
newborn | O
infant | O
. | O

Since we need to build a one-to-one correspondence between tokens and classes, we can train a rightmost **many-to-many** neural network model from this picture:

![Image showing common recurrent neural network patterns.](../17-GenerativeNetworks/images/unreasonable-effectiveness-of-rnn.jpg)

> *Image from [this blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) by [Andrej Karpathy](http://karpathy.github.io/). NER token classification models correspond to the right-most network architecture on this picture.*

## Training NER models

Since a NER model is essentially a token classification model, we can use RNNs that we are already familiar with for this task. In this case, each block of recurrent network will return the token ID. The following example notebook shows how to train LSTM for token classification.

## ✍️ Example Notebooks: NER

Continue your learning in the following notebook:

* [NER with TensorFlow](NER-TF.ipynb)

## Conclusion

A NER model is a **token classification model**, which means that it can be used to perform token classification. This is a very common task in NLP, helping to recognize specific entities within text including places, names, dates, and more.

## 🚀 Challenge

Complete the assignment linked below to train a named entity recognition model for medical terms, then try it on a different dataset.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ai/quiz/38)

## Review & Self Study

Read through the blog [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) and follow along with the Further Reading section in that article to deepen your knowledge.

## [Assignment](lab/README.md)

In the assignment for this lesson, you will have to train a medical entity recognition model. You can start with training an LSTM model as described in this lesson, and proceed with using the BERT transformer model. Read [the instructions](lab/README.md) to get all the details.

## Applied lab: token-budget preflight (R38-T041)

This lab shows how a prompt is checked against a token budget before it is sent
to a model, and how truncation is reported. Example and tests:
[applied-labs/r38-t041](applied-labs/r38-t041/example.py).

**Prerequisites:** Python 3 only (standard library). No model download, paid
provider or GPU.

**Simulation label:** the tokenizer is a whitespace splitter, a teaching
approximation, not the model's real tokenizer. Real tokenizers split into
subwords, so real counts will differ. The lab makes no claim about real
hardware, cloud behaviour or trained-model performance.

**Run (from the repository root):**

    python3 lessons/5-NLP/19-NER/applied-labs/r38-t041/example.py "one two three four" 3

**Fixture / input:** text `one two three four`, budget `3`

**Expected output:**

    Tokenizer: whitespace (teaching approximation, not the model's real tokenizer)
    Budget: 3
    Retained: [one,two,three]
    Removed: 1 token(s) [four]

An empty input retains no tokens; a negative budget is rejected with an error.

**Run the tests:**

    python3 -m unittest discover -s lessons/5-NLP/19-NER/applied-labs/r38-t041 -p "test_*.py"
