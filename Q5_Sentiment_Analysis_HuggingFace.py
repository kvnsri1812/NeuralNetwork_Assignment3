from transformers import pipeline

# Load pre-trained sentiment analysis pipeline
classifier = pipeline("sentiment-analysis")

# Input sentence
sentence = "Despite the high price, the performance of the new MacBook is outstanding."

# Get result
result = classifier(sentence)[0]

# Print output
print(f"Sentiment: {result['label']}")
print(f"Confidence Score: {result['score']:.4f}")
