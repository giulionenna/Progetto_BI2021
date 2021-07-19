

# %%
import pandas as pd
data=pd.read_excel(r'data_dec.xlsx')
df=pd.DataFrame(data)

df = df.drop(columns='langdetect')
df.loc[df['unsure_wrong_detection']==True].shape #controllo dei tweet la cui lingua non è stata riconosciuta
df = df.loc[df['unsure_wrong_detection']==False] #filtraggio dei tweet non riconosciuti

### Suddivisione dei Dataset per lingua ###

#en
df_en = df.loc[df['lang_detect_final']=='en']
df_en = df_en.drop(columns=['lang_detect_final', 'unsure_wrong_detection'])

#es
df_es = df.loc[df['lang_detect_final']=='es']
df_es = df_es.drop(columns=['lang_detect_final', 'unsure_wrong_detection'])

#fr
df_fr = df.loc[df['lang_detect_final']=='fr']
df_fr = df_fr.drop(columns=['lang_detect_final', 'unsure_wrong_detection'])
df_en=pd.DataFrame(pd.read_excel('df_en_clustered.xlsx'))

# %%
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
porter = PorterStemmer()

df_en.text_clean=df_en.text_clean.astype('str')

text_stem = []
#df_en['text_clean']=df_en['text_clean'].astype('str')

for twt in df_en.text_clean:
    tweet = twt.lower()
    token_words = word_tokenize(tweet) #tokenizzazione
    token_filter_words = [w for w in token_words if not w in stop_words] #stopword filtering
    stem_sentence=[]
    for word in token_filter_words:
        stem_sentence.append(porter.stem(word)) #stemming
        stem_sentence.append(' ')
    
    text_stem.append(''.join(stem_sentence))

df_en['text_stem']=text_stem

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(data, mindf, maxdf=1,  lan='english'):

    tfidfVectorizer = TfidfVectorizer(min_df=mindf, max_df=maxdf,
                                    stop_words=lan)
    tfidf_matrix = tfidfVectorizer.fit_transform(data['text_stem'])

    print("sono state prodotte", len(tfidfVectorizer.get_feature_names()), "parole nel processo di tf-idf con mindf=", mindf)
    ## Conversione TfIdf in Dataframe ##

    # tfidfVectorizer ritorna una matrice sparsa che non permette la visualizzazione agevole della matrice TfIdf

    tf_idf = pd.DataFrame(columns=tfidfVectorizer.get_feature_names(), index=data['text_clean'], dtype=float)
    
    M=tfidf_matrix.todense()
    for i in range(0, data.shape[0]):
        tf_idf.iloc[i, :]=M[i,:]

    ## classifica delle parole
    #score={}
    #for word in tfidfVectorizer.get_feature_names():
    #    score[word]= tf_idf_en[word].mean()

    #score_sroted = dict(sorted(score.items(), key=lambda item: item[1], reverse=True))
    #print("Score delle prime 10 parole: ", list(score_sroted.items())[0:10])

    return tf_idf, M






# %%
df_en_school=pd.DataFrame(pd.read_excel('df_en_cluster_schools.xlsx'))
df_en_texas=pd.DataFrame(pd.read_excel('df_en_cluster_texas.xlsx'))
df_en_vax=pd.DataFrame(pd.read_excel('df_en_cluster_vax.xlsx'))

# %%
#genero una matrice one-hot encoded a partire dalla tf-idf completa

def onehot(r):
    if r==0:
        return 0
    else:
        return 1

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

def rules_fp_growth(X, min_sup=0.005):
    freq_itemsets = fpgrowth(X, min_sup, use_colnames=True)
    rules = association_rules(freq_itemsets, metric='confidence', min_threshold=0.3)
    return rules

def rules(df, minsup=0.01):
    M, tmp = tfidf(df, 0, maxdf=0.7)
    M=M.applymap(onehot)
    rules=rules_fp_growth(M, min_sup=minsup)
    return rules

# %%

rules(df_en_school).to_excel('rules_school.xlsx')
rules(df_en_texas, minsup=0.02).to_excel('rules_texas.xlsx')
rules(df_en_vax, minsup=0.02).to_excel('rules_vax.xlsx')


# %%
M_school=M_school.reset_index(drop=True)
M_school

# %% [markdown]
# ## Esaminazione dei cluster

# %%
#visualizzazione dei centroidi tramite PCA

M = model.cluster_centers_

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#Standardizzo i dati
scaler = StandardScaler()
M = scaler.fit_transform(M)

#Applico la PCA mantenendo le prime 3 componenti
pca=PCA(n_components=3)
M = pca.fit_transform(M)

#scatter 3d dei centroidi prodotti
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

figure = pyplot.figure()
ax = Axes3D(figure)

ax.scatter(M[:,0], M[:,1], M[:,2])
pca.explained_variance_

pyplot.show()

# %% [markdown]
# # CODICE SPERIMENTALE - IGNORA
# %% [markdown]
# # Utilizo del DB-SCAN per raggruppare i cluster

# %%
from sklearn.cluster import DBSCAN
M= model.cluster_centers_
pca=PCA(0.95)
M= pca.fit_transform(scaler.fit_transform(M))

clustering = DBSCAN(eps=1, metric='cosine').fit(model.cluster_centers_)
clustering.labels_

# %% [markdown]
# # DB Scan Clustering

# %%
scaler = StandardScaler()
from sklearn.neighbors import NearestNeighbors
M=scaler.fit_transform(tfidf_en_matrix_00)

nbrs = NearestNeighbors(n_neighbors=5, radius=20, n_jobs=-1).fit(M)
distances, indices = nbrs.kneighbors(M)
distances_plot = [d[4] for d in distances ]
#print(indices[0],distances[0])


# %%
distances_plot.sort()
plt.plot(range(M.shape[0]), distances_plot)


# %%
from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=100, min_samples=5).fit(M)


df_en['labels_DBSCAN']= clustering.labels_
df_en.loc[df_en['labels_DBSCAN']==0]



# %%
df_en.loc[df_en['labels_DBSCAN']==-1]


# %%
cloud(df_en, 'labels_DBSCAN', -1)


# %%
tf_idf_en.reset_index(drop=True, inplace=True )
df_en.reset_index(drop=True, inplace=True)
df_en_concat = pd.concat([df_en, tf_idf_en], axis=1)
df_en_concat.to_excel('df_en_concat.xlsx')


# %%
tf_idf_en.to_excel('tf_idf_en.xlsx')


