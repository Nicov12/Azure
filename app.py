from flask import Flask , jsonify, request
import pandas as pd
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import spacy
#import es_dep_news_trf
from collections import Counter
from string import punctuation

# import sklearn
# print(sklearn.__version__)

#nlp = spacy.load("es_dep_news_trf")
nlp = spacy.load("es_core_news_sm")
#nlp = es_dep_news_trf.load()

# Funciones manejadoras de las recomendaciones #

#Dejando los documentos en un formato listo para pasarlo al modelo W2V
def formattingCorpus(documents):
    corpus = []
    for words in documents:
        corpus.append(words.split())
    return corpus
#print(corpus[:5])

# Construcción del recomendador #

# Creación del vector avgword2vec, que contiene el promedio de los valores de las palabras de cada documento
def vectors(model, corpus):
    
    # Lista para guardar los vectores (documento a vector)
    global word_embeddings
    word_embeddings = []

    # Obteniendo el vector de cada documento
    for line in corpus:
        avgword2vec = None
        count = 0
        for word in line:
            if word in model.wv.index_to_key:
                count += 1
                if avgword2vec is None:
                    avgword2vec = model.wv[word]
                else:
                    avgword2vec = avgword2vec + model.wv[word]
                    
                
        if avgword2vec is not None:
            avgword2vec = avgword2vec / count
        
            word_embeddings.append(avgword2vec)

# Función de recomendación
def recommendations(title, model, corpus, df):
    
    # Invocando la función que asigna los vectores a cada documento
    vectors(model, corpus)
    
    # Similaridad del coseno de los vectores obtenidos

    cosine_similarities = cosine_similarity(word_embeddings, word_embeddings)

    # Tomando el título de cada libro
    books = df[['Título', 'URL']]
    # Mapeo inverso del indice de los libros
    indices = pd.Series(df.index, index = df['Título']).drop_duplicates()
         
    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    recommend = books.iloc[book_indices]
    return recommend

# Función de recomendación con un input nuevo
def recommendationsAdapted(model, corpus, df):
    
    # Invocando la función que asigna los vectores a cada documento
    vectors(model, corpus)
    
    # Similaridad del coseno de los vectores obtenidos

    cosine_similarities = cosine_similarity(word_embeddings, word_embeddings)

    # Tomando el título de cada libro
    books = df[['Título', 'URL']]
    # Mapeo inverso del indice de los libros
    indices = pd.Series(df.index, index = df['Título']).drop_duplicates()
         
    #idx = indices[title]
    idx = 10000 # Indice del nuevo documento ingresado (entra de último)
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)
    sim_scores = sim_scores[1:10]
    book_indices = [i[0] for i in sim_scores]
    recommend = books.iloc[book_indices]
    return recommend

# Creación y entretamiento del modelo W2V
def buildModelW2V(documents):
    #model = Word2Vec.load("word2vec.model")
    model = Word2Vec(vector_size = 300, window=5, min_count = 2, workers = -1)
    model.build_vocab(formattingCorpus(documents))
    model.train(formattingCorpus(documents), total_examples=model.corpus_count, epochs = 5)
    return model

# Procesamiento del input ingresado al chat por el usuario
def get_hotwords(text):
    result = []
    pos_tag = ['PROPN', 'NOUN'] 
    doc = nlp(text.lower()) 
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    return result

def processingInput(input):
    output = set(get_hotwords(input))
    most_common_list = Counter(output).most_common(10)
    input_processed = ""
    for i in range(len(most_common_list)):
        if i < len(most_common_list)-1:
            input_processed += most_common_list[i][0] + " "
        else:
            input_processed += most_common_list[i][0]
    return input_processed

# ----- Fin funciones manejadoras de las recomendaciones ----- # 

# Desde aquí es la definición de los servicios, no es necesario implementar nada de la ejecución
app = Flask(__name__)

cors = CORS(app, resources={r"/getRec/*": {"origins": "https://pepechatbot.netlify.app"}})
@app.route('/', methods=['GET'])
def home():
    return "Books API"

#@cross_origin()
@app.route('/getRec', methods=['POST']) # Endpoint que hace la recomendación dada una entrada del usuario
def gettingRec():
    input = request.json
    user_input = input['input'] # Input del usuario
    new_element = pd.Series([user_input])
    df = pd.read_csv("libros_keywords_v1.2.csv", sep=";")
    documents = df["Clean documents"]
    documents = documents.append(new_element, ignore_index=True)
    documents = documents.values.astype('U')
    print("Entrada del usuario:", input['input'])
    model = buildModelW2V(documents) # Construcción del modelo W2V
    print("Recomendaciones:")
    recomendaciones = recommendationsAdapted(model , formattingCorpus(documents), df).values.astype('U')
    print(recomendaciones)
    return jsonify([{'rec1' : recomendaciones[0][0], 'url' : recomendaciones[0][1]}, {'rec2' : recomendaciones[1][0], 'url' : recomendaciones[1][1]},
                    {'rec3' : recomendaciones[2][0], 'url' : recomendaciones[2][1]}, {'rec4' : recomendaciones[3][0], 'url' : recomendaciones[3][1]},
                    {'rec5' : recomendaciones[4][0], 'url' : recomendaciones[4][1]}])
    #return jsonify({'msg' : 'Success', 'input' : input['input']})

@app.route('/getRecV2', methods=['POST']) # Endpoint que hace la recomendación dado un título de un libro
def gettingRecV2():
    input = request.json
    user_input = input['input'] # Input del usuario
    #new_element = pd.Series([user_input])
    df = pd.read_csv("libros_keywords_v1.2.csv", sep=";")
    documents = df["Clean documents"]
    # documents = documents.append(new_element, ignore_index=True)
    # documents = documents.values.astype('U')
    print("Entrada del usuario:", input['input'])
    model = buildModelW2V(documents) # Construcción del modelo W2V
    print("Recomendaciones:")
    recomendaciones = recommendations(user_input, model , formattingCorpus(documents), df).values.astype('U')
    print(recomendaciones)
    return jsonify([{'rec1' : recomendaciones[0][0], 'url' : recomendaciones[0][1]}, {'rec2' : recomendaciones[1][0], 'url' : recomendaciones[1][1]},
                    {'rec3' : recomendaciones[2][0], 'url' : recomendaciones[2][1]}, {'rec4' : recomendaciones[3][0], 'url' : recomendaciones[3][1]},
                    {'rec5' : recomendaciones[4][0], 'url' : recomendaciones[4][1]}])

@app.route('/getRecV3', methods=['POST']) # Endpoint que hace la recomendación dado input personalizado en el chat de la aplicación
def gettingRecV3():
    input = request.json
    user_input = processingInput(input['input']) # Input del usuario
    new_element = pd.Series([user_input])
    df = pd.read_csv("libros_keywords_v1.2.csv", sep=";")
    documents = df["Clean documents"]
    documents = documents.append(new_element, ignore_index=True)
    documents = documents.values.astype('U')
    print("Entrada del usuario:", user_input)
    model = buildModelW2V(documents) # Construcción del modelo W2V
    print("Recomendaciones:")
    recomendaciones = recommendationsAdapted(model , formattingCorpus(documents), df).values.astype('U')
    print(recomendaciones)
    return jsonify([{'rec1' : recomendaciones[0][0], 'url' : recomendaciones[0][1]}, {'rec2' : recomendaciones[1][0], 'url' : recomendaciones[1][1]},
                    {'rec3' : recomendaciones[2][0], 'url' : recomendaciones[2][1]}, {'rec4' : recomendaciones[3][0], 'url' : recomendaciones[3][1]},
                    {'rec5' : recomendaciones[4][0], 'url' : recomendaciones[4][1]}])