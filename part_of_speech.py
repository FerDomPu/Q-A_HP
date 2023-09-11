
def look_for_entity(question):
    from flair.data import Sentence
    from flair.models import SequenceTagger
    nouns = []
    # Load tagger
    tagger = SequenceTagger.load("flair/upos-english")
    # Create question sentence
    sentence = Sentence(question)
    # Predict NER tags
    tagger.predict(sentence)
    # Get the nouns of the question
    for word in sentence:
        if word.tag == "NOUN":
            nouns.append(word.text)
    return nouns
