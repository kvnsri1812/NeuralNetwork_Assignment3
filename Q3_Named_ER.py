import spacy

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

# Input sentence
text = "Barack Obama served as the 44th President of the United States and won the Nobel Peace Prize in 2009."

# Process the text
doc = nlp(text)

# Print each entity with its label and position
for ent in doc.ents:
    print(f"Entity: {ent.text}")
    print(f"Label: {ent.label_}")
    print(f"Start: {ent.start_char}, End: {ent.end_char}\n")