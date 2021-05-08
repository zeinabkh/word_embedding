from parsivar import  Normalizer,Tokenizer, FindStems
from gensim.models import Word2Vec,TfidfModel, LdaModel
from gensim.corpora import Dictionary
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score, normalized_mutual_info_score as NMI
from sklearn.decomposition import TruncatedSVD
from matplotlib import pyplot as plt, gridspec
news_labels = {
    "ادب و هنر":0,
    "سیاسی":1,
     "اقتصاد":2,
     "ورزش":3,
    "اجتماعی":4
}
import math
news_file = open("Hamshahri.txt", encoding="utf8")
news = news_file.read().split("\n")
news_list = []
stop_words = ['که', 'از', 'به', 'در', 'برای', 'زیرا', 'همچنین', 'آن', 'این', 'و', 'شد', 'شو', 'کرد', 'است', 'هست',
              'را', 'با', 'نیست', 'ای', 'الا', 'اما', 'اگر', 'می', 'خود', 'ای', 'نیز', 'وی', 'هم', 'ما', 'نمی',
              'پیش', 'همه', 'بی', 'من'
    , 'چه', 'هیچ', 'ولی', 'حتی', 'توسط', 'شما', 'تو', 'او', 'ایشان', 'هنوز', 'البته', 'فقط', 'شاید', 'شان', 'روی',
              'مانند', 'کجا', 'کی', 'چطور', 'چگونه', 'مگر', 'چندین', 'کدام', 'چیزی', 'چیز', 'دیگر', 'دیگری', 'مثل',
              'بلی', 'همین']
save_file = open("data.txt", 'w', encoding="utf8")
for i, n in enumerate(news):
    doc = []
    for word in Tokenizer().tokenize_words(n[n.index("@") + 11]):
        doc.append(word)
    news_list.append({"content":doc,"label": news_labels[n[0:n.index("@")]]})



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
def representation_of_doc_1(c,news_list,word_vec):
    docs_vector = []
    for doc in news_list:
        vector = np.zeros(c)
        for word in doc["content"]:
            try:
                vector += word_vec[word]
            except KeyError:
                continue
        vector /= len(doc["content"])
        docs_vector.append(vector)
    return docs_vector


# represent  documents as vectors that are construct from the average of word vectors in each document use TF.
def representation_of_doc_2(c,docs,dictionary,BoW_corpus,word_vec):
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
                # print(word_vec[dictionary[word_tfidf[0]]])
                vector += word_tfidf[1] * np.array(word_vec[dictionary[word_tfidf[0]]])
                count += word_tfidf[1]
            except KeyError:
                continue
            except TypeError:
                print(type(word_vec[dictionary[word_tfidf[0]]]),word_vec[dictionary[word_tfidf[0]]])
        docs_vector.append(vector/count)
    return docs_vector
# load_data("Hamshahri.txt")
def hamshari_doc_vec_process(path):
    word_vec = {}
    file_reader = open(path,encoding="utf-8")
    data_stream = file_reader.read().split("\n")
    for w in data_stream[1:]:
      word = w.split(" ")
      if len(word) > 0:
        word_vec[word[0]] = [float(v) for v in word[1:] if len(v) > 0]
    return word_vec

def SVD_term_docs(docs,dictionary,BoW_corpus):
   doc_word_matrix = np.zeros((len(BoW_corpus),dictionary))
   for i,docs in enumerate(BoW_corpus):
       for words in docs:
           doc_word_matrix[i,words[0]] = words[1]
   # d, c, w = np.linalg.svd(doc_word_matrix)
   svd = TruncatedSVD(n_components=300)
   return svd.fit_transform(doc_word_matrix.transpose())
   # print(d)


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
      print("f1_score micro: ", f1_score(true_labels, custom_labels, average='micro'))
      print("f1_score: macro", f1_score(true_labels, custom_labels, average='macro'))
      print("NMI", NMI(true_labels, predict_label))

def Lda_topic_model(docs,dictionary,nb_topics,true_labels):
    k = 5
    lda = LdaModel(docs, num_topics=k, id2word=dictionary, passes=10)

    top_words = [[word[::-1] for word, _ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _, beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    nb_words = 12
    f, ax = plt.subplots(3,2,figsize=(20,15))
    for i in range(nb_topics):
        # ax = plt.subplot(gs[i])
        m,n = np.unravel_index(i,shape=(3,2))[0],np.unravel_index(i,shape=(3,2))[1]
        ax[m,n].barh(range(nb_words), top_betas[i][:nb_words], align='center', color='green', ecolor='black')
        ax[m,n].invert_yaxis()
        ax[m,n].set_yticks(range(nb_words))
        ax[m,n].set_yticklabels(top_words[i][:nb_words])
        ax[m,n].set_title("Topic " + str(i))
    plt.show()
    # get distribution of docs on topics.
    dist_on_topics = lda.get_document_topics(docs)
    topic_predict = []
    for d in dist_on_topics:
        p = 0
        win_topic = 0
        print(d)
        for i,t in enumerate(d):
            if t[1]> p:
               p= t[1]
               win_topic = t[0]
        print(win_topic)
        topic_predict.append(win_topic)
    mat = confusion_matrix(true_labels, topic_predict)
    print(mat)
    cluster_to_class = {}
    for i in range(5):
        cluster_to_class[i] = np.argmax(mat[:, i])
    custom_labels = [cluster_to_class[c] for c in topic_predict]
    print("accuracy:", accuracy_score(true_labels, custom_labels))
    print("f1_score micro: ", f1_score(true_labels, custom_labels, average='micro'))
    print("f1_score: macro", f1_score(true_labels, custom_labels, average='macro'))
    print("NMI", NMI(true_labels, custom_labels))



# news_list = load_data("Hamshahri.txt")
docs = list(map(lambda x: x['content'], news_list))
labels = list(map(lambda x: x['label'], news_list))
# print(docs)
dictionary = Dictionary(docs)
BoW_corpus = [dictionary.doc2bow(text) for text in docs]
skip_embedding = skip_gram(news_list)
# clustering with average vector.

print("average vectors")
docs_vectors_average = representation_of_doc_1(300, news_list, skip_embedding.wv)
kmean_clustering(docs_vectors_average, labels)
# print(docs_vectors_average)
# # clustering with average vector by tf_idf.
print("average vectors by tf_idf")
docs_vectors_tfidf = representation_of_doc_2(300, docs, dictionary, BoW_corpus, skip_embedding.wv)
kmean_clustering(docs_vectors_tfidf, labels)
# clustering with svd.
# print("svd")
# docs_vectors_svd = SVD_term_docs(docs,len(dictionary),BoW_corpus)
# kmean_clustering(docs_vectors_svd, labels)
# hamshahri doc vec
print("hamshahri vectors.")
hamshari_word_vec = hamshari_doc_vec_process("G:\master_matus\99_2\\NLP\HWS\hamshahri.fa.text.300.vec")
docs_vectors_tfidf1 = representation_of_doc_2(300, docs, dictionary, BoW_corpus, hamshari_word_vec)
kmean_clustering(docs_vectors_tfidf1, labels)
Lda_topic_model(BoW_corpus,dictionary,5,labels)
