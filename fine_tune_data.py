import pandas as pd

data_names = pd.read_csv("Characters_data.csv", usecols=["Name"])
names_list = data_names["Name"].tolist()
names_split = []
names_complete = []
names_total = []

# Create a list with names and tag
for name in names_list:
    name_splited_list = []
    for i in range(len(name.split())):
        if i == 0:
            name1 = name.split()[i]
            ner_tag = "B-PER"
            name_individual_list = [name1,"0","0",ner_tag]
            name_splited_list.append(name_individual_list)
        else:
            name1 = name.split()[i]
            ner_tag = "I-PER"
            name_individual_list = [name1,"0","0",ner_tag]
            name_splited_list.append(name_individual_list)
    names_complete.append(name_splited_list)

# Split the data into training and testing
from sklearn.model_selection import train_test_split
import random

train_data, test_data = train_test_split(names_complete, test_size=0.2, random_state=random.seed())

print("Conjunto de entrenamiento:")
print(train_data)
print("\nConjunto de prueba:")
print(test_data)

# Create text documents with data sets
with open("data/train.txt","w") as file:
    for name in train_data:
        for entity in name:
            file.write(" ".join(entity)+"\n")
        file.write("\n")
        
with open("data/test.txt","w") as file:
    for name in test_data:
        for entity in name:
            file.write(" ".join(entity)+"\n")
        file.write("\n")