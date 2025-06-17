import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

# Input sentence
sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."

# 1. Tokenization
tokens = word_tokenize(sentence)

# 2. Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# 3. Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

# Output
print("Original Tokens:", tokens)
print("Tokens Without Stopwords:", filtered_tokens)
print("Stemmed Words:", stemmed_tokens)