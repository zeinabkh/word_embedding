from parsivar import  Normalizer,Tokenizer, FindStems
from gensim.models import Word2Vec,TfidfModel
from gensim.corpora import Dictionary
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score
news_labels = {
    "ادب و هنر":0,
    "سیاسی":1,
     "اقتصاد":2,
     "ورزش":3,
    "اجتماعی":4
}
import math
# news_file = open("Hamshahri.txt", encoding="utf8")
# news = news_file.read().split("\n")
# news_list = []
stop_words = ['که', 'از', 'به', 'در', 'برای', 'زیرا', 'همچنین', 'آن', 'این', 'و', 'شد', 'شو', 'کرد', 'است', 'هست',
              'را', 'با', 'نیست', 'ای', 'الا', 'اما', 'اگر', 'می', 'خود', 'ای', 'نیز', 'وی', 'هم', 'ما', 'نمی',
              'پیش', 'همه', 'بی', 'من'
    , 'چه', 'هیچ', 'ولی', 'حتی', 'توسط', 'شما', 'تو', 'او', 'ایشان', 'هنوز', 'البته', 'فقط', 'شاید', 'شان', 'روی',
              'مانند', 'کجا', 'کی', 'چطور', 'چگونه', 'مگر', 'چندین', 'کدام', 'چیزی', 'چیز', 'دیگر', 'دیگری', 'مثل',
              'بلی', 'همین']
# save_file = open("data.txt", 'w', encoding="utf8")
# for i, n in enumerate(news):
#     doc = ""
#     for word in Tokenizer().tokenize_words(Normalizer().normalize(doc_string=n[n.index("@") + 11:])):
#         doc += FindStems().convert_to_stem(word) + "*"
#     doc += n[0:n.index("@")]
#     save_file.write(doc + "\n")
#     print(i)
# save_file.close()


# load_data, read docs file and every file conside as a item by content and label
# input: path if data file
# output: return the list that contain docs and their labels.
def load_data(path):
    news_file = open("data.txt", encoding="utf8")
    news = news_file.read().split("\n")
    # print(news)
    news_list = []
    for i, n in enumerate(news):
        if len(n) > 1:
            words = n.split("*")
            news_list.append({
                "content": [word
                            for word in words[0:-1] if word not in stop_words],
                "label": news_labels[words[-1]]
            })
    return news_list

# get word embedding by skip gram.
# use gensim.word2vec


def skip_gram(news_list):
    docs_list = list(map(lambda x: x['content'], news_list))
    skip_gram = Word2Vec(docs_list,min_count=1,vector_size=300, workers=3, window =5, sg = 1)
    return skip_gram

# represent  documents as vectors that are construct from the average of word vectors in each document.
def representation_of_doc_1(c,news_list,skip_embedding):
    docs_vector = []
    for doc in news_list:
        vector = np.zeros(c)
        for word in doc["content"]:
            try:
                vector += skip_embedding.wv[word]
            except KeyError:
                continue
        vector /= len(doc["content"])
        docs_vector.append(vector)
    return docs_vector


# represent  documents as vectors that are construct from the average of word vectors in each document use TF.
def representation_of_doc_2(c,docs,dictionary,BoW_corpus,skip_embedding):
    tfidf = TfidfModel(BoW_corpus)
    docs_vector = []
    print(len(BoW_corpus))
    for doc in BoW_corpus:
        tf_idf_vec = tfidf[doc]
        # print(tf_idf_vec)
        vector = np.zeros(c)
        count = 0
        for word_tfidf in tf_idf_vec:
            try:
                vector += word_tfidf[1] * skip_embedding.wv[dictionary[word_tfidf[0]]]
                count += word_tfidf[1]
            except KeyError:
                continue
        docs_vector.append(vector/count)
    return docs_vector
# load_data("Hamshahri.txt")


def SVD_term_docs(docs,dictionary,BoW_corpus):
   doc_word_matrix = np.zeros((len(BoW_corpus),dictionary))
   for i,docs in enumerate(BoW_corpus):
       for words in docs:
           doc_word_matrix[i,words[0]] = words[1]
   d, c, w = np.linalg.svd(doc_word_matrix)
   print(d)


def kmean_clustering(doc_vectors, true_labels):
      k_mean = KMeans(n_clusters=5)
      k_mean.fit(doc_vectors)
      predict_label = k_mean.labels_
      mat = confusion_matrix(true_labels, predict_label)
      print(mat)
      cluster_to_class = {}
      for i in range(5):
           cluster_to_class[i] = np.argmax(mat[:, i])
      custom_labels = [cluster_to_class[c] for c in predict_label]
      print("accuracy:", accuracy_score(true_labels, custom_labels))
      print("f1_score: ", f1_score(true_labels, custom_labels, average='micro'))


news_list = load_data("Hamshahri.txt")
docs = list(map(lambda x: x['content'], news_list))
labels = list(map(lambda x: x['label'], news_list))
# print(docs)
dictionary = Dictionary(docs)
BoW_corpus = [dictionary.doc2bow(text) for text in docs]
skip_embedding = skip_gram(news_list)
# docs_vectors_average = representation_of_doc_1(300, news_list, skip_embedding)
# docs_vectors_tfidf = representation_of_doc_2(300, docs, dictionary, BoW_corpus, skip_embedding)
# kmean_clustering(docs_vectors_average, labels)
# kmean_clustering(docs_vectors_tfidf, labels)
SVD_term_docs(docs,len(dictionary),BoW_corpus)