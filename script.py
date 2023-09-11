import pandas as pd
from tokenization_classification_flair import look_for_character
from part_of_speech import look_for_entity
from word_embeddings import look_for_embeddings
import logging

# Disable Flair's informational messages
logging.getLogger('flair').setLevel(logging.ERROR)

# Load the database
data = pd.read_csv("Characters_data.csv")
data = data.drop(["Id"],axis=1)
data["Gender"] = data["Gender"].fillna("Male")
data["Job"] = data["Job"].fillna("No job")
data["House"] = data["House"].fillna("Out of school")
data.fillna("No info", inplace=True)
data = data.astype("str")
data.columns = [column.lower() for column in data.columns]

# Get the columns to see what topics you can ask the table about
topics = data.columns

# Example question
question = "What is Hagrid's job?"

# Find the question character
characters = look_for_character(question)
# if characters == []:
#     print("No se ha encontrado un character")
# else:
#     for character in characters:
#         print(f"El personaje elegido es: {character}")


# Build the regular expression to find partial matches
characters_num = len(characters)
patron_cualquiera = ".*"
for character in characters:
    name_num = len(character.split())
    expresion_regular = patron_cualquiera.join(nombre for nombre in character.split())
    expresion_regular = patron_cualquiera + expresion_regular + patron_cualquiera

# Apply the filter using the regular expression and assign the result to a new DataFrame
data_characters = data[data['name'].str.fullmatch(expresion_regular, case=False)]
# print(data_characters)

# Find the question topic
question_nouns = look_for_entity(question)
# print(question_nouns)
column_question = look_for_embeddings(topics,question_nouns)
# print(column_question)

# Print response
response = data_characters[column_question].iloc[0]
print(response)
