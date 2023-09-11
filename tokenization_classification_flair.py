
# Function
def look_for_character(question):
    from flair.data import Sentence
    from flair.models import SequenceTagger
    # Load the model
    model = SequenceTagger.load("resources/taggers/conllpp_harry/best-model.pt")
    # Make a sentence
    question = Sentence(question)
    # Predict NER tags
    model.predict(question)
    character = []
    question[0]
    # Get person name
    for entity in question.get_spans("ner"):
        if entity.tag == "PER":
            character.append(entity.text)
    return character

