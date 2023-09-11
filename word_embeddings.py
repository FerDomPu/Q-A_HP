
def look_for_embeddings(columns,nouns):
    from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
    from flair.data import Sentence
    import torch
    import numpy as np
    columns_embeddings = list()
    # Create a StackedEmbedding object that combines glove and forward/backward flair embeddings
    stacked_embeddings = StackedEmbeddings([
                                            WordEmbeddings('glove'),
                                            FlairEmbeddings('news-forward'),
                                            FlairEmbeddings('news-backward'),
                                            ])
    # Embed columns
    for column in columns:
        sentence = Sentence(column)
        stacked_embeddings.embed(sentence)
        for token in sentence:
            token_embedding = token.embedding
            columns_embeddings.append([column,token_embedding])
            
    nouns_similarity = []
    nouns_prueba = []
    
    # Compare the nouns obtained with the columns and find the maximum similarity    
    for noun in nouns:
        sentence = Sentence(noun)
        stacked_embeddings.embed(sentence)
        embedding_noun = sentence[0].embedding
        max_similarity = 0
        for columns_embeddings_split in columns_embeddings:
            similarity = torch.nn.functional.cosine_similarity(embedding_noun,columns_embeddings_split[1],dim=0)
            if similarity > max_similarity:
                max_column = columns_embeddings_split[0]
                max_similarity = similarity.item()
            nouns_prueba.append([noun,columns_embeddings_split[0],similarity.item()])
        nouns_similarity.append([noun,max_column,max_similarity])
    nouns_similarity = np.array(nouns_similarity)
    index_max_similarity = np.argmax(nouns_similarity[:,2])
    column_max_similarity = nouns_similarity[index_max_similarity][1]
    
    return column_max_similarity
