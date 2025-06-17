# **CS5720 Home Assignment 3 – RNN, NLP, Attention & Transformers**

 **Course Title**: CS5720 – Neural Networks and Deep Learning
 
 **University**: University of Central Missouri
 
 **Term**: Summer 2025
##   

**Student Name**: Komatlapalli Venkata Naga Sri

**Student ID**: 700773763

---

### **Assignment Overview**

This assignment covers key tasks in Natural Language Processing (NLP) and sequence modeling. It is divided into five focused parts:

Q1. LSTM-based text generation using temperature scaling
Q2. NLP preprocessing: tokenization, stopword removal, and stemming
Q3. Named Entity Recognition (NER) with SpaCy
Q4. Manual implementation of scaled dot-product attention
Q5. Sentiment analysis using HuggingFace Transformers

Each part uses Python along with libraries such as TensorFlow, NLTK, SpaCy, NumPy, and Transformers.

---

## **Part 1: LSTM-Based Text Generation**

### **Objective:**

To train an LSTM model on character-level text and explore how temperature scaling affects output randomness.

### **What Was Done:**

* Loaded *Shakespeare* dataset.
* Converted text into integer sequences using a character index mapping.
* Built an LSTM model using TensorFlow with:

  * Embedding layer
  * LSTM layer
  * Dense output layer
* Trained the model for one epoch.
* Generated sample text using different temperature values: 0.5, 1.0, 1.5.

### **Outcome:**

* Lower temperature (0.5) gave repetitive and grammatically correct output.
* Higher temperature (1.5) produced more diverse but sometimes less coherent text.
* Temperature scaling was applied using: `predictions / temperature` before softmax.

---

## **Part 2: NLP Preprocessing Pipeline**

### **Objective:**

To perform basic preprocessing on a sentence using tokenization, stopword removal, and stemming.

### **What Was Done:**

* Input sentence: `"NLP techniques are used in virtual assistants like Alexa and Siri."`
* Tokenized the sentence using NLTK’s `word_tokenize`.
* Removed English stopwords using NLTK’s built-in list.
* Applied stemming using the `PorterStemmer`.

### **Output:**

1. Original Tokens
2. Tokens without stopwords
3. Stemmed versions

### **Short Answers:**

* **Stemming vs Lemmatization**: Stemming trims suffixes (e.g., “running” → “run”), while lemmatization returns accurate base forms using grammar.
* **Stopword Removal**: Useful in keyword extraction, but harmful in sentiment tasks where words like “not” carry meaning.

---

## **Part 3: Named Entity Recognition with SpaCy**

### **Objective:**

To extract named entities from a given sentence and understand their labels and positions.

### **What Was Done:**

* Input sentence:
  `"Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."`
* Used SpaCy’s `en_core_web_sm` model.
* Printed each entity’s text, label, and character positions.

### **Output:**

Identified entities included `Barack Obama (PERSON)`, `United States (GPE)`, `Nobel Peace Prize (WORK_OF_ART)`, `2009 (DATE)`.

### **Short Answers:**

* **NER vs POS Tagging**: NER finds real-world names; POS tagging identifies grammatical roles.
* **Applications**: Used in financial document analysis, search engine understanding, and chatbots.

---

## **Part 4: Scaled Dot-Product Attention**

### **Objective:**

To manually implement the scaled dot-product attention mechanism.

### **What Was Done:**

* Defined Q, K, V matrices as:

  ```python
  Q = [[1, 0, 1, 0], [0, 1, 0, 1]]
  K = [[1, 0, 1, 0], [0, 1, 0, 1]]
  V = [[1, 2, 3, 4], [5, 6, 7, 8]]
  ```
* Computed Q·Kᵀ, scaled by √d, applied softmax, and multiplied by V.

### **Results:**

* Printed attention weights and final output matrix.
* Showed how attention focuses more on similar query-key pairs.

### **Short Answers:**

* **Why divide by √d?**: To avoid large softmax values and stabilize training.
* **Benefit of Self-Attention**: Captures context by letting each word attend to others in a sentence.

---

## **Part 5: Sentiment Analysis Using HuggingFace Transformers**

### **Objective:**

To use a pre-trained sentiment classification model to analyze a sentence.

### **What Was Done:**

* Input sentence:
  `"Despite the high price, the performance of the new MacBook is outstanding."`
* Used HuggingFace’s `pipeline("sentiment-analysis")`.
* Printed sentiment label and confidence score.

### **Output:**

* Sentiment: **POSITIVE**
* Confidence Score: \~**0.9987**

### **Short Answers:**

* **BERT vs GPT**: BERT is encoder-based for understanding; GPT is decoder-based for generation.
* **Why Use Pretrained Models**: Saves time, generalizes better, and works well with small datasets.
