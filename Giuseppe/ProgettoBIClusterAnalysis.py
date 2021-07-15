# IMPORTO LIBRERIE
import pandas as pd
import numpy as np
import nltk as nltk

data=pd.read_excel('C:/Users/gsppp/OneDrive/Poli/Quarto Anno/Business Intelligence per Big Data/Progetto/.csv/data_dec.xlsx')
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

df_en.to_excel('dataen.xlsx')

# PREPROCESSING

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

# TFIDF

from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(data, mindf,  lan='english'):

    tfidfVectorizer = TfidfVectorizer(min_df=mindf,
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

tf_idf_en, tfidf_en_matrix=tfidf(df_en, 0)
tf_idf_en, tfidf_en_matrix=tfidf(df_en, 0.001)
tf_idf_en, tfidf_en_matrix=tfidf(df_en, 0.005)
tf_idf_en, tfidf_en_matrix=tfidf(df_en, 0.01)
tf_idf_en, tfidf_en_matrix=tfidf(df_en, 0.05)

tf_idf_en_01, tfidf_en_matrix_01 = tfidf(df_en, 0.001)
tf_idf_en_05, tfidf_en_matrix_05 = tfidf(df_en, 0.005)
tf_idf_en_10, tfidf_en_matrix_10 = tfidf(df_en, 0.01)

tf_idf_en_00, tfidf_en_matrix_00 = tfidf(df_en, 0)

tfidf_en_matrix_01

# PCA 

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def feature_redux(M):
    original_size=M.shape[1]
    #Standardizzo i dati
    scaler = StandardScaler()
    M_redux = scaler.fit_transform(M)

    #Applico la PCA mantenendo il 95% della varianza
    pca=PCA(0.90)
    M_redux = pca.fit_transform(M_redux)

    print("Il numero di feature è stato ridotto da ",original_size,"a", pca.n_components_)

    #scatter 3d dei punti prodotti
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot

    figure = pyplot.figure()
    ax = Axes3D(figure)

    ax.scatter(M_redux[:,0], M_redux[:,1], M_redux[:,2])
    pca.explained_variance_
    pyplot.show()

    return M

tfidf_en_matrix_PCA = feature_redux(tfidf_en_matrix_00)

# SSD PLOT

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

def kmeans_ssd_plot(M):
    SSD = []
    K=range(2,20)
    t=time.perf_counter()
    for k in K:
        print("Kmeans clustering con k = ", k)
        km=KMeans(n_clusters=k, max_iter=300, n_init=20,random_state=3425)
        km = km.fit(M)
        SSD.append(km.inertia_)
    print("valutazione del kmeans clustering terminata in", time.perf_counter()-t)
    plt.plot(K, SSD, 'bx-')

kmeans_ssd_plot(tfidf_en_matrix_PCA)
kmeans_ssd_plot(tfidf_en_matrix_05)
kmeans_ssd_plot(tfidf_en_matrix_01)
kmeans_ssd_plot(tfidf_en_matrix_10)

k=15

model = KMeans(n_clusters=k, max_iter=300, n_init=20,random_state=3425)
model.fit(tfidf_en_matrix_01)
df_en['label_kmeans_01']=model.labels_

model = KMeans(n_clusters=k, max_iter=300, n_init=20,random_state=3425)
model.fit(tfidf_en_matrix_05)
df_en['label_kmeans_05']=model.labels_

model = KMeans(n_clusters=k, max_iter=300, n_init=20,random_state=3425)
model.fit(tfidf_en_matrix_10)
df_en['label_kmeans_10']=model.labels_

model = KMeans(n_clusters=k, max_iter=300, n_init=20,random_state=3425)
model.fit(tfidf_en_matrix_PCA)
df_en['label_kmeans_PCA']=model.labels_

df_en.to_excel('clustered_data_KMeans.xlsx')

# VALUTAZIONE QUALITATIVA CLUSTER

from wordcloud import WordCloud
def cloud_OLD(df, label, ncluster, text_index='text_clean'):

    frequent_words=['texas']
    df_0 = df.loc[df[label]==ncluster]
    text = df_0[text_index].str.cat(sep='')
    text = text.lower()
    text=' '.join([word for word in text.split() if not word in frequent_words])
    wordcloud = WordCloud(max_font_size=300, max_words=100, background_color="white", width=1920, height=1080).generate(text)
    fig1 = plt.figure(figsize=(6,6))
    fig1.set_size_inches(18, 7)
    plt.imshow(wordcloud)
    print(text)

from wordcloud import WordCloud
def cloud(df, label, ncluster, text_index='text_clean'):

    frequent_words=['covid', 're']
    

    df_0 = df.loc[df[label]==ncluster]
    text = []
    for twt in df_0[text_index]:
        token_words = word_tokenize(twt) #tokenizzazione
        token_filter_words_1 = [w for w in token_words if not w in stop_words]
        token_filter_words_2 = [w for w in token_filter_words_1 if not w in frequent_words] #stopword filtering
        for w in token_filter_words_2:
            text.append(w)
    
    
    txt=' '.join([word for word in text])
    wordcloud = WordCloud(max_font_size=300, max_words=100, background_color="white", width=1920, height=1080).generate(txt)
    fig1 = plt.figure(figsize=(6,6))
    fig1.set_size_inches(18, 7)
    plt.imshow(wordcloud)

for i in range(k):
    cloud(df_en, 'label_kmeans_01', i)

# ANALISI CLUSTER TEXAS (4), VACCINES (2), SCHOOL (14)
# Comparo la sentiment analysis dei cluster 2, 4 e 14 divisi rispetto ai giorni per vedere
# se ci sono delle variazioni. Usiamp solo i valori trovati con min_df = 0.01 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer

data = pd.read_excel('C:/Users/gsppp/OneDrive/Poli/Quarto Anno/Business Intelligence per Big Data/Progetto/ProgettoBIClusterAnalysis/clustered_data_KMeans.xlsx')
df_en=pd.DataFrame(data)

# Creo i dataframe che userò per le analisi
TexasDF  = df_en.loc[df_en['label_kmeans_01'] == 4].drop(['label_kmeans_05','label_kmeans_10','label_kmeans_PCA'],axis=1)
PharmaDF = df_en.loc[df_en['label_kmeans_01'] == 2].drop(['label_kmeans_05','label_kmeans_10','label_kmeans_PCA'],axis=1)
SchoolDF = df_en.loc[df_en['label_kmeans_01'] == 14].drop(['label_kmeans_05','label_kmeans_10','label_kmeans_PCA'],axis=1)

# Oggetto per svolgere la SA
sia = SentimentIntensityAnalyzer()

# Questa funzione mi serve per etichettare i vari tweet
def labeler(val):
    if val > 0.05:
        return "positive"
    elif val < -0.05:
        return "negative"
    else:
        return "neutral"

writer = pd.ExcelWriter('Polarity', engine='xlsxwriter')

# Calcolo i tweet positivi, negativi e neutri per il DF sul Texas
TexasDF['positive'] = [sia.polarity_scores(str(tweet))['pos'] for tweet in TexasDF.text_clean]
TexasDF['negative'] = [sia.polarity_scores(str(tweet))['neg'] for tweet in TexasDF.text_clean]
TexasDF['neutral'] = [sia.polarity_scores(str(tweet))['neu'] for tweet in TexasDF.text_clean]
TexasDF['compound'] = [sia.polarity_scores(str(tweet))['compound'] for tweet in TexasDF.text_clean]
TexasDF['overall score'] = [labeler(val) for val in TexasDF.compound]

TexasCNT = [0,0,0]
TexasCNT[0] = TexasDF['overall score'].values.tolist().count("positive")
TexasCNT[1] = TexasDF['overall score'].values.tolist().count("negative")
TexasCNT[2] = TexasDF['overall score'].values.tolist().count("neutral")

TexasCNT
len(TexasDF.compound) == sum(TexasCNT)

# Donut plot - Texas
TexasLabels = 'positive','negative','neutral'
Colors = ['#2F329F','#2F7998','#2D475F'] # blue, turchese, grigio
plt.pie(TexasCNT,labels=TexasLabels,autopct='%1.1f%%',pctdistance=0.5,colors=Colors)
plt.title('Sentiment analysis del cluster relativo al Texas')
plt.axis('equal')
plt.legend()
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

TexasDF.to_excel(writer, sheet_name='TexasDF')

# Calcolo i tweet positivi, negativi e neutri per il DF sulle aziende farmaceutiche
PharmaDF['positive'] = [sia.polarity_scores(str(tweet))['pos'] for tweet in PharmaDF.text_clean]
PharmaDF['negative'] = [sia.polarity_scores(str(tweet))['neg'] for tweet in PharmaDF.text_clean]
PharmaDF['neutral'] = [sia.polarity_scores(str(tweet))['neu'] for tweet in PharmaDF.text_clean]
PharmaDF['compound'] = [sia.polarity_scores(str(tweet))['compound'] for tweet in PharmaDF.text_clean]
PharmaDF['overall score'] = [labeler(val) for val in PharmaDF.compound]

PharmaCNT = [0,0,0]
PharmaCNT[0] = PharmaDF['overall score'].values.tolist().count("positive")
PharmaCNT[1] = PharmaDF['overall score'].values.tolist().count("negative")
PharmaCNT[2] = PharmaDF['overall score'].values.tolist().count("neutral")

PharmaCNT
len(PharmaDF.compound) == sum(PharmaCNT)

# Donut plot - vaccini
PharmaLabels = 'positive','negative','neutral'
Colors = ['#2F329F','#2F7998','#2D475F'] # blue, turchese, grigio
plt.pie(PharmaCNT,labels=PharmaLabels,autopct='%1.1f%%',pctdistance=0.5,colors=Colors)
plt.title('Sentiment analysis del cluster relativo ai vaccini')
plt.axis('equal')
plt.legend()
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()

PharmaDF.to_excel(writer, sheet_name='PharmaDF')

# Calcolo i tweet positivi, negativi e neutri per il DF sulle scuole
SchoolDF['positive'] = [sia.polarity_scores(str(tweet))['pos'] for tweet in SchoolDF.text_clean]
SchoolDF['negative'] = [sia.polarity_scores(str(tweet))['neg'] for tweet in SchoolDF.text_clean]
SchoolDF['neutral'] = [sia.polarity_scores(str(tweet))['neu'] for tweet in SchoolDF.text_clean]
SchoolDF['compound'] = [sia.polarity_scores(str(tweet))['compound'] for tweet in SchoolDF.text_clean]
SchoolDF['overall score'] = [labeler(val) for val in SchoolDF.compound]

SchoolCNT = [0,0,0]
SchoolCNT[0] = SchoolDF['overall score'].values.tolist().count("positive")
SchoolCNT[1] = SchoolDF['overall score'].values.tolist().count("negative")
SchoolCNT[2] = SchoolDF['overall score'].values.tolist().count("neutral")

SchoolCNT
len(SchoolDF.compound) == sum(SchoolCNT)

# Donut plot - scuole
SchoolLabels = 'positive','negative','neutral'
Colors = ['#2F329F','#2F7998','#2D475F'] # blue, turchese, grigio
plt.pie(SchoolCNT,labels=SchoolLabels,autopct='%1.1f%%',pctdistance=0.5,colors=Colors)
plt.title('Sentiment analysis del cluster relativo alla scuola')
plt.axis('equal')
plt.legend()
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


SchoolDF.to_excel(writer, sheet_name='SchoolDF')
writer.save()
writer.close()

# FILTRO I TWEET RISPETTO ALLE DATE DI PUBBLICAZIONE E FACCIO DI NUOVO SA PER VEDERE COME VARIANO
# LE REAZIONI NEL TEMPO
from datetime import datetime

# Calcolo i tweet positivi, negativi e neutri per il DF sul Texas rispetto alla data di pubblicazione del tweet
TexasPosDF = pd.read_excel('Polarity.xlsx','TexasDF')
TexasPosDF = TexasPosDF.loc[TexasPosDF['overall score']=='positive'].sort_values(by='created_at_ntz')
TexasPosDF = TexasPosDF[~(TexasPosDF['created_at_ntz'] <= '2021-02-28')]
TexasPosDF['new_date'] = [d.date() for d in TexasPosDF['created_at_ntz']]
TexasPosDF['new_date'] = TexasPosDF['new_date'].astype("datetime64")

TexasNegDF = pd.read_excel('Polarity.xlsx','TexasDF')
TexasNegDF = TexasNegDF.loc[TexasNegDF['overall score']=='negative'].sort_values(by='created_at_ntz')
TexasNegDF = TexasNegDF[~(TexasNegDF['created_at_ntz'] <= '2021-02-28')]
TexasNegDF['new_date'] = [d.date() for d in TexasNegDF['created_at_ntz']]
TexasNegDF['new_date'] = TexasNegDF['new_date'].astype("datetime64")

TexasNeuDF = pd.read_excel('Polarity.xlsx','TexasDF')
TexasNeuDF = TexasNeuDF.loc[TexasNeuDF['overall score']=='neutral'].sort_values(by='created_at_ntz')
TexasNeuDF = TexasNeuDF[~(TexasNeuDF['created_at_ntz'] <= '2021-02-28')]
TexasNeuDF['new_date'] = [d.date() for d in TexasNeuDF['created_at_ntz']]
TexasNeuDF['new_date'] = TexasNeuDF['new_date'].astype("datetime64")

TexasPosDF.groupby('new_date').size()
TexasNegDF.groupby('new_date').size()
TexasNeuDF.groupby('new_date').size()

# # Stacked Barplot - Texas: data di pubblicazione
width = 0.35       
fig, ax = plt.subplots()
labels = ['1 mar','2 mar','3 mar','4 mar','5 mar','6 mar','7 mar','8 mar']
ax.bar(labels, TexasPosDF.groupby('new_date').size(), width, label='Pos', color = '#2F329F', alpha = 0.5)
ax.bar(labels, TexasNegDF.groupby('new_date').size(), width, label='Neg', color = '#2F7998', alpha = 0.5, bottom=TexasPosDF.groupby('new_date').size())
ax.set_ylabel('Date')
ax.set_ylabel('Frequenza assoluta')
ax.set_title('Sentiment analysis del cluster relativo al Texas\n rispetto al giorno di creazione del tweet')
ax.legend()
plt.show()

# Calcolo i tweet positivi, negativi e neutri per il DF sui vaccini rispetto alla data di pubblicazione del tweet
PharmaPosDF = pd.read_excel('Polarity.xlsx','PharmaDF')
PharmaPosDF = PharmaPosDF.loc[PharmaPosDF['overall score']=='positive'].sort_values(by='created_at_ntz')
PharmaPosDF = PharmaPosDF[~(PharmaPosDF['created_at_ntz'] <= '2021-02-28')]
PharmaPosDF['new_date'] = [d.date() for d in PharmaPosDF['created_at_ntz']]
PharmaPosDF['new_date'] = PharmaPosDF['new_date'].astype("datetime64")

PharmaNegDF = pd.read_excel('Polarity.xlsx','PharmaDF')
PharmaNegDF = PharmaNegDF.loc[PharmaNegDF['overall score']=='negative'].sort_values(by='created_at_ntz')
PharmaNegDF = PharmaNegDF[~(PharmaNegDF['created_at_ntz'] <= '2021-02-28')]
PharmaNegDF['new_date'] = [d.date() for d in PharmaNegDF['created_at_ntz']]
PharmaNegDF['new_date'] = PharmaNegDF['new_date'].astype("datetime64")

PharmaNeuDF = pd.read_excel('Polarity.xlsx','PharmaDF')
PharmaNeuDF = PharmaNeuDF.loc[PharmaNeuDF['overall score']=='neutral'].sort_values(by='created_at_ntz')
PharmaNeuDF = PharmaNeuDF[~(PharmaNeuDF['created_at_ntz'] <= '2021-02-28')]
PharmaNeuDF['new_date'] = [d.date() for d in PharmaNeuDF['created_at_ntz']]
PharmaNeuDF['new_date'] = PharmaNeuDF['new_date'].astype("datetime64")

PhPos = PharmaPosDF.groupby('new_date').size()
PhNeg = PharmaNegDF.groupby('new_date').size()
PharmaNeuDF.groupby('new_date').size()

TMP = PhNeg
PhNeg = PhPos
i=0
while i < len(PhPos):
    if i < 5:
        PhNeg[i] = TMP[i]
    elif i == 5:
        PhNeg[i] = 0
    elif i == 6:
        PhNeg[i] = TMP[i-1]
    i = i+1

PhPos = PharmaPosDF.groupby('new_date').size()

# Stacked Barplot - vaccini: data di pubblicazione
width = 0.35       
fig, ax = plt.subplots()
labels = ['1 mar','2 mar','3 mar','4 mar','5 mar','6 mar','7 mar']
ax.bar(labels, PhPos, width, label='Pos', color = '#2F329F', alpha = 0.5)
ax.bar(labels, PhNeg, width, label='Neg', color = '#2F7998', alpha = 0.5, bottom=PhPos)
ax.set_ylabel('Date')
ax.set_ylabel('Frequenza assoluta')
ax.set_title('Sentiment analysis del cluster relativo ai vaccini\n rispetto al giorno di creazione del tweet')
ax.legend()
plt.show()


# Calcolo i tweet positivi, negativi e neutri per il DF sulla scuola rispetto alla data di pubblicazione del tweet
SchoolPosDF = pd.read_excel('Polarity.xlsx','SchoolDF')
SchoolPosDF = SchoolPosDF.loc[SchoolPosDF['overall score']=='positive'].sort_values(by='created_at_ntz')
SchoolPosDF = SchoolPosDF[~(SchoolPosDF['created_at_ntz'] <= '2021-02-28')]
SchoolPosDF['new_date'] = [d.date() for d in SchoolPosDF['created_at_ntz']]
SchoolPosDF['new_date'] = SchoolPosDF['new_date'].astype("datetime64")

SchoolNegDF = pd.read_excel('Polarity.xlsx','SchoolDF')
SchoolNegDF = SchoolNegDF.loc[SchoolNegDF['overall score']=='negative'].sort_values(by='created_at_ntz')
SchoolNegDF = SchoolNegDF[~(SchoolNegDF['created_at_ntz'] <= '2021-02-28')]
SchoolNegDF['new_date'] = [d.date() for d in SchoolNegDF['created_at_ntz']]
SchoolNegDF['new_date'] = SchoolNegDF['new_date'].astype("datetime64")

SchoolNeuDF = pd.read_excel('Polarity.xlsx','SchoolDF')
SchoolNeuDF = SchoolNeuDF.loc[SchoolNeuDF['overall score']=='neutral'].sort_values(by='created_at_ntz')
SchoolNeuDF = SchoolNeuDF[~(SchoolNeuDF['created_at_ntz'] <= '2021-02-28')]
SchoolNeuDF['new_date'] = [d.date() for d in SchoolNeuDF['created_at_ntz']]
SchoolNeuDF['new_date'] = SchoolNeuDF['new_date'].astype("datetime64")

ScPos = SchoolPosDF.groupby('new_date').size()
ScNeg = SchoolNegDF.groupby('new_date').size()
SchoolNeuDF.groupby('new_date').size()

TMP = ScNeg
ScNeg = ScPos
i = 0
while i < len(ScPos):
    if i < 7:
        ScNeg[i] = TMP[i]
    if i == 7:
        ScNeg[i] = 0
    i = i+1

ScPos = SchoolPosDF.groupby('new_date').size()

# # Stacked Barplot - scuole: data di pubblicazione
width = 0.35       
fig, ax = plt.subplots()
labels = ['1 mar','2 mar','3 mar','4 mar','5 mar','6 mar','7 mar','8 mar']
ax.bar(labels, ScPos, width, label='Pos', color = '#2F329F', alpha = 0.5)
ax.bar(labels, ScNeg, width, label='Neg', color = '#2F7998', alpha = 0.5, bottom=ScPos)
ax.set_ylabel('Date')
ax.set_ylabel('Frequenza assoluta')
ax.set_title('Sentiment analysis del cluster relativo alle scuole\n rispetto al giorno di creazione del tweet')
ax.legend()
plt.show()

# FILTRO I TWEET RISPETTO ALL'ORA DI PUBBLICAZIONE E FACCIO DI NUOVO SA PER VEDERE COME VARIANO
# LE REAZIONI NEL TEMPO

def counter(series):
    cnt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for line in series:
        cnt[int(line)] = cnt[int(line)] + 1
    return cnt

# Calcolo i tweet positivi, negativi e neutri per il DF sul Texas rispetto all'ora di pubblicazione del tweet
TexasHourDF = pd.read_excel('Polarity.xlsx','TexasDF')
TexasHourDF['hour'] = pd.to_datetime(TexasHourDF['created_at_ntz'],format = '%H:%M:%S').dt.hour
TexasHourDF = TexasHourDF.sort_values(by='hour')

TexasHourPos = TexasHourDF.loc[TexasHourDF['overall score']=='positive']
TexasHourNeg = TexasHourDF.loc[TexasHourDF['overall score']=='negative']
TexasHourNeu = TexasHourDF.loc[TexasHourDF['overall score']=='neutral']

TxPosHour = counter(TexasHourPos['hour'])
TxNegHour = counter(TexasHourNeg['hour'])

# Stacked Barplot - Texas: ora di pubblicazione
width = 0.35       
fig, ax = plt.subplots()
labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
ax.bar(labels, TxPosHour, width, label='Pos', color = '#2F329F', alpha = 0.5)
ax.bar(labels, TxNegHour, width, label='Neg', color = '#2F7998', alpha = 0.5, bottom=TxPosHour)
ax.set_ylabel('Ore del giorno')
ax.set_ylabel('Frequenza assoluta')
ax.set_title('Sentiment analysis del cluster relativo al Texas\n rispetto alle ore del giorno')
ax.legend()
plt.show()

# Calcolo i tweet positivi, negativi e neutri per il DF sui vaccini rispetto all'ora di pubblicazione del tweet
PharmaHourDF = pd.read_excel('Polarity.xlsx','PharmaDF')
PharmaHourDF['hour'] = pd.to_datetime(PharmaHourDF['created_at_ntz'],format = '%H:%M:%S').dt.hour
PharmaHourDF = PharmaHourDF.sort_values(by='hour')

PharmaHourPos = PharmaHourDF.loc[PharmaHourDF['overall score']=='positive']
PharmaHourNeg = PharmaHourDF.loc[PharmaHourDF['overall score']=='negative']
PharmaHourNeu = PharmaHourDF.loc[PharmaHourDF['overall score']=='neutral']

PhPosHour = counter(PharmaHourPos['hour'])
PhNegHour = counter(PharmaHourNeg['hour'])

# Stacked Barplot - vaccini: ora di pubblicazione
width = 0.35       
fig, ax = plt.subplots()
labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
ax.bar(labels, PhPosHour, width, label='Pos', color = '#2F329F', alpha = 0.5)
ax.bar(labels, PhNegHour, width, label='Neg', color = '#2F7998', alpha = 0.5, bottom=PhPosHour)
ax.set_ylabel('Ore del giorno')
ax.set_ylabel('Frequenza assoluta')
ax.set_title('Sentiment analysis del cluster relativo ai vaccini\n rispetto alle ore del giorno')
ax.legend()
plt.show()

# Calcolo i tweet positivi, negativi e neutri per il DF sulla scuola rispetto all'ora di pubblicazione del tweet
SchoolHourDF = pd.read_excel('Polarity.xlsx','SchoolDF')
SchoolHourDF['hour'] = pd.to_datetime(SchoolHourDF['created_at_ntz'],format = '%H:%M:%S').dt.hour
SchoolHourDF = SchoolHourDF.sort_values(by='hour')

SchoolHourPos = SchoolHourDF.loc[SchoolHourDF['overall score']=='positive']
SchoolHourNeg = SchoolHourDF.loc[SchoolHourDF['overall score']=='negative']
SchoolHourNeu = SchoolHourDF.loc[SchoolHourDF['overall score']=='neutral']

ScPosHour = counter(SchoolHourPos['hour'])
ScNegHour = counter(SchoolHourNeg['hour'])

# Stacked Barplot - scuole: ora di pubblicazione
width = 0.35       
fig, ax = plt.subplots()
labels = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
ax.bar(labels, ScPosHour, width, label='Pos', color = '#2F329F', alpha = 0.5)
ax.bar(labels, ScNegHour, width, label='Neg', color = '#2F7998', alpha = 0.5, bottom=ScPosHour)
ax.set_ylabel('Ore del giorno')
ax.set_ylabel('Frequenza assoluta')
ax.set_title('Sentiment analysis del cluster relativo alle scuole\n rispetto alle ore del giorno')
ax.legend()
plt.show()

# SENTIMENT ANALYSIS RELATIVA AI LINK
import pandas as pd
import numpy as np
import nltk as nltk
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

# Questa funzione mi serve per etichettare i vari tweet
def labeler(val):
    if val > 0.05:
        return "positive"
    elif val < -0.05:
        return "negative"
    else:
        return "neutral"

def linkCounter(series):
    cnt = np.zeros(6,dtype=int)
    for line in series:
        if int(line) <= 5:
            cnt[int(line)] = cnt[int(line)] + 1
    return cnt

sia = SentimentIntensityAnalyzer()

data = pd.read_excel('C:/Users/gsppp/OneDrive/Poli/Quarto Anno/Business Intelligence per Big Data/Progetto/ProgettoBIClusterAnalysis/data_dec.xlsx')
linkDF=pd.DataFrame(data)

linkDF['positive'] = [sia.polarity_scores(str(tweet))['pos'] for tweet in linkDF.text_clean]
linkDF['negative'] = [sia.polarity_scores(str(tweet))['neg'] for tweet in linkDF.text_clean]
linkDF['neutral'] = [sia.polarity_scores(str(tweet))['neu'] for tweet in linkDF.text_clean]
linkDF['compound'] = [sia.polarity_scores(str(tweet))['compound'] for tweet in linkDF.text_clean]
linkDF['overall score'] = [labeler(val) for val in linkDF.compound]


# Per semplicità consideriamo solo i tweet contentnenti al più 5 link. I tweet con più di 5 link sono 3 su un totale di oltre 4000 tweet e perciò sono
# trascurabili.

linkPosDF = linkDF.loc[linkDF['overall score']=='positive'].sort_values(by='links')['links']
linkNegDF = linkDF.loc[linkDF['overall score']=='negative'].sort_values(by='links')['links']

cntPosLinks = linkCounter(linkPosDF)
cntNegLinks = linkCounter(linkNegDF)

width = 0.35       
fig, ax = plt.subplots()
labels = ['00','01','02','03','04','05']
ax.bar(labels, cntPosLinks, width, label='Pos', color = '#2F329F', alpha = 0.5)
ax.bar(labels, cntNegLinks, width, label='Neg', color = '#2F7998', alpha = 0.5, bottom=cntPosLinks)
ax.set_ylabel('Numero di link nel tweet')
ax.set_ylabel('Frequenza assoluta')
ax.set_title('Sentiment analysis rispetto\n al numero di link nel tweet')
ax.legend()
plt.show()

# SENTIMENT ANALYSIS RELATIVA AI "MI PIACE"
import pandas as pd
import numpy as np
import nltk as nltk
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

def labeler(val):
    if val > 0.05:
        return "positive"
    elif val < -0.05:
        return "negative"
    else:
        return "neutral"

def favCounter(series):
    cnt = np.zeros(max(series)-min(series)+1,dtype=int)
    for num in series:
        cnt[num] = cnt[num] + 1
    return cnt

def shrinker(series):
    histCnt = np.zeros(12,dtype=int)
    i = 0
    for num in series:
        if i == 0:
            histCnt[i] = histCnt[i] + num
        elif i > 0 and i <= 10:
            histCnt[0] = histCnt[0] + num
        elif i > 10 and i <= 20:
            histCnt[1] = histCnt[1] + num
        elif i > 20 and i <= 50:
            histCnt[2] = histCnt[2] + num
        elif i > 50 and i <= 100:
            histCnt[3] = histCnt[3] + num
        elif i > 100 and i <= 200:
            histCnt[4] = histCnt[4] + num
        elif i > 200 and i <= 500:
            histCnt[5] = histCnt[5] + num
        elif i > 500 and i <= 1000:
            histCnt[6] = histCnt[6] + num
        elif i > 1000 and i <= 2000:
            histCnt[7] = histCnt[7] + num
        elif i > 2000 and i <= 5000:
            histCnt[8] = histCnt[8] + num
        elif i > 5000 and i <= 10000:
            histCnt[9] = histCnt[9] + num
        else:
            histCnt[10] = histCnt[10] + num
        i = i+1
    return histCnt

sia = SentimentIntensityAnalyzer()

# Texas

dataTexas = pd.read_excel('C:/Users/gsppp/OneDrive/Poli/Quarto Anno/Business Intelligence per Big Data/Progetto/ProgettoBIClusterAnalysis/Polarity.xlsx',sheet_name=0)
fav_countDFTexas=pd.DataFrame(dataTexas)

fav_countDFTexas['positive'] = [sia.polarity_scores(str(tweet))['pos'] for tweet in fav_countDFTexas.text_clean]
fav_countDFTexas['negative'] = [sia.polarity_scores(str(tweet))['neg'] for tweet in fav_countDFTexas.text_clean]
fav_countDFTexas['neutral'] = [sia.polarity_scores(str(tweet))['neu'] for tweet in fav_countDFTexas.text_clean]
fav_countDFTexas['compound'] = [sia.polarity_scores(str(tweet))['compound'] for tweet in fav_countDFTexas.text_clean]
fav_countDFTexas['overall score'] = [labeler(val) for val in fav_countDFTexas.compound]

fav_countPosDFTexas = fav_countDFTexas.loc[fav_countDFTexas['overall score']=='positive'].sort_values(by='favorite_count')['favorite_count']
fav_countNegDFTexas = fav_countDFTexas.loc[fav_countDFTexas['overall score']=='negative'].sort_values(by='favorite_count')['favorite_count']

# Per chiarezza i conteggi dei "mi piaci" li riportiamo in scala di log2
favPosCntTexas = shrinker(favCounter(fav_countPosDFTexas))
i = 0
for num in favPosCntTexas:
    if num != 0:
        favPosCntTexas[i] = np.log2(favPosCntTexas[i])
    i = i+1

favNegCntTexas = shrinker(favCounter(fav_countNegDFTexas))
i = 0
for num in favNegCntDFTexas:
    if num != 0:
        favNegCntTexas[i] = np.log2(favNegCntTexas[i])
    i = i+1
 

width = 0.35       
fig, ax = plt.subplots()
labels = ['0','0-10','10-20','20-50','50-100','100-200','200-500','500-1000','1000-2000','2000-5000','5000-10000','>10000']
ax.bar(labels, favPosCntTexas, width, label='Pos', color = '#2F329F', alpha = 0.5)
ax.bar(labels, favNegCntTexas, width, label='Neg', color = '#2F7998', alpha = 0.5, bottom=favPosCntTexas)
plt.xticks(rotation=45)
ax.set_ylabel('Numero di link nel tweet')
ax.set_ylabel('Frequenza assoluta in scala del logaritmo di 2')
ax.set_title('Sentiment analysis rispetto al numero di "mi piace"\n dei tweet sulle restrizioni in Texas')
ax.legend()
plt.show()

# Pharma

dataPharma = pd.read_excel('C:/Users/gsppp/OneDrive/Poli/Quarto Anno/Business Intelligence per Big Data/Progetto/ProgettoBIClusterAnalysis/Polarity.xlsx',sheet_name=1)
fav_countDFPharma=pd.DataFrame(dataPharma)

fav_countDFPharma['positive'] = [sia.polarity_scores(str(tweet))['pos'] for tweet in fav_countDFPharma.text_clean]
fav_countDFPharma['negative'] = [sia.polarity_scores(str(tweet))['neg'] for tweet in fav_countDFPharma.text_clean]
fav_countDFPharma['neutral'] = [sia.polarity_scores(str(tweet))['neu'] for tweet in fav_countDFPharma.text_clean]
fav_countDFPharma['compound'] = [sia.polarity_scores(str(tweet))['compound'] for tweet in fav_countDFPharma.text_clean]
fav_countDFPharma['overall score'] = [labeler(val) for val in fav_countDFPharma.compound]

fav_countPosDFPharma = fav_countDFPharma.loc[fav_countDFPharma['overall score']=='positive'].sort_values(by='favorite_count')['favorite_count']
fav_countNegDFPharma = fav_countDFPharma.loc[fav_countDFPharma['overall score']=='negative'].sort_values(by='favorite_count')['favorite_count']

# Per chiarezza i conteggi dei "mi piaci" li riportiamo in scala di log2
favPosCntPharma = shrinker(favCounter(fav_countPosDFPharma))
i = 0
for num in favPosCntPharma:
    if num != 0:
        favPosCntPharma[i] = np.log2(favPosCntPharma[i])
    i = i+1

favNegCntPharma = shrinker(favCounter(fav_countNegDFPharma))
i = 0
for num in favPosCntPharma:
    if num != 0:
        favPosCntPharma[i] = np.log2(favPosCntPharma[i])
    i = i+1
 

width = 0.35       
fig, ax = plt.subplots()
labels = ['0','0-10','10-20','20-50','50-100','100-200','200-500','500-1000','1000-2000','2000-5000','5000-10000','>10000']
ax.bar(labels, favPosCntPharma, width, label='Pos', color = '#2F329F', alpha = 0.5)
ax.bar(labels, favNegCntPharma, width, label='Neg', color = '#2F7998', alpha = 0.5, bottom=favPosCntPharma)
plt.xticks(rotation=45)
ax.set_ylabel('Numero di link nel tweet')
ax.set_ylabel('Frequenza assoluta in scala del logaritmo di 2')
ax.set_title('Sentiment analysis rispetto\n al numero di "mi piace" dei tweet sui vaccini')
ax.legend()
plt.show()

# School

dataSchool = pd.read_excel('C:/Users/gsppp/OneDrive/Poli/Quarto Anno/Business Intelligence per Big Data/Progetto/ProgettoBIClusterAnalysis/Polarity.xlsx',sheet_name=2)
fav_countDFSchool=pd.DataFrame(dataSchool)

fav_countDFSchool['positive'] = [sia.polarity_scores(str(tweet))['pos'] for tweet in fav_countDFSchool.text_clean]
fav_countDFSchool['negative'] = [sia.polarity_scores(str(tweet))['neg'] for tweet in fav_countDFSchool.text_clean]
fav_countDFSchool['neutral'] = [sia.polarity_scores(str(tweet))['neu'] for tweet in fav_countDFSchool.text_clean]
fav_countDFSchool['compound'] = [sia.polarity_scores(str(tweet))['compound'] for tweet in fav_countDFSchool.text_clean]
fav_countDFSchool['overall score'] = [labeler(val) for val in fav_countDFSchool.compound]

fav_countPosDFSchool = fav_countDFSchool.loc[fav_countDFSchool['overall score']=='positive'].sort_values(by='favorite_count')['favorite_count']
fav_countNegDFSchool = fav_countDFSchool.loc[fav_countDFSchool['overall score']=='negative'].sort_values(by='favorite_count')['favorite_count']

# Per chiarezza i conteggi dei "mi piaci" li riportiamo in scala di log2
favPosCntSchool = shrinker(favCounter(fav_countPosDFSchool))
i = 0
for num in favPosCntSchool:
    if num != 0:
        favPosCntSchool[i] = np.log2(favPosCntSchool[i])
    i = i+1

favNegCntSchool = shrinker(favCounter(fav_countNegDFSchool))
i = 0
for num in favNegCntSchool:
    if num != 0:
        favNegCntSchool[i] = np.log2(favNegCntSchool[i])
    i = i+1

width = 0.35       
fig, ax = plt.subplots()
labels = ['0','0-10','10-20','20-50','50-100','100-200','200-500','500-1000','1000-2000','2000-5000','5000-10000','>10000']
ax.bar(labels, favPosCntSchool, width, label='Pos', color = '#2F329F', alpha = 0.5)
ax.bar(labels, favNegCntSchool, width, label='Neg', color = '#2F7998', alpha = 0.5, bottom=favPosCntSchool)
plt.xticks(rotation=45)
ax.set_ylabel('Numero di link nel tweet')
ax.set_ylabel('Frequenza assoluta in scala del logaritmo di 2')
ax.set_title('Sentiment analysis rispetto\n al numero di "mi piace" del tweet sulla scuola')
ax.legend()
plt.show()

# SENTIMENT ANALYSIS RELATIVA AI RETWEET
import pandas as pd
import numpy as np
import nltk as nltk
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer

def labeler(val):
    if val > 0.05:
        return "positive"
    elif val < -0.05:
        return "negative"
    else:
        return "neutral"

def RTCounter(series):
    cnt = np.zeros(max(series)-min(series)+1,dtype=int)
    for num in series:
        cnt[num] = cnt[num] + 1
    return cnt

def RTshrinker(series):
    histCnt = np.zeros(11,dtype=int)
    i = 0
    for num in series:
        if i == 0:
            histCnt[0] = histCnt[0] + num
        elif i > 0 and i <= 10:
            histCnt[1] = histCnt[1] + num
        elif i > 10 and i <= 20:
            histCnt[2] = histCnt[2] + num
        elif i > 20 and i <= 50:
            histCnt[3] = histCnt[3] + num
        elif i > 50 and i <= 100:
            histCnt[4] = histCnt[4] + num
        elif i > 100 and i <= 200:
            histCnt[5] = histCnt[5] + num
        elif i > 200 and i <= 500:
            histCnt[6] = histCnt[6] + num
        elif i > 500 and i <= 1000:
            histCnt[7] = histCnt[7] + num
        elif i > 1000 and i <= 2000:
            histCnt[8] = histCnt[8] + num
        elif i > 2000 and i <= 5000:
            histCnt[9] = histCnt[9] + num
        else:
            histCnt[10] = histCnt[10] + num
        i = i+1
    return histCnt

# Texas

dataTexas = pd.read_excel('C:/Users/gsppp/OneDrive/Poli/Quarto Anno/Business Intelligence per Big Data/Progetto/ProgettoBIClusterAnalysis/Polarity.xlsx',sheet_name=0)
RT_countDFTexas=pd.DataFrame(dataTexas)

RT_countDFTexas['positive'] = [sia.polarity_scores(str(tweet))['pos'] for tweet in RT_countDFTexas.text_clean]
RT_countDFTexas['negative'] = [sia.polarity_scores(str(tweet))['neg'] for tweet in RT_countDFTexas.text_clean]
RT_countDFTexas['neutral'] = [sia.polarity_scores(str(tweet))['neu'] for tweet in RT_countDFTexas.text_clean]
RT_countDFTexas['compound'] = [sia.polarity_scores(str(tweet))['compound'] for tweet in RT_countDFTexas.text_clean]
RT_countDFTexas['overall score'] = [labeler(val) for val in RT_countDFTexas.compound]

RT_countPosDFTexas = RT_countDFTexas.loc[RT_countDFTexas['overall score']=='positive'].sort_values(by='favorite_count')['favorite_count']
RT_countNegDFTexas = RT_countDFTexas.loc[RT_countDFTexas['overall score']=='negative'].sort_values(by='favorite_count')['favorite_count']

# Per chiarezza i conteggi dei "mi piaci" li riportiamo in scala di log2
RTPosCntTexas = RTshrinker(RTCounter(RT_countPosDFTexas))
RTNegCntTexas = RTshrinker(RTCounter(RT_countNegDFTexas))

i = 0
for num in RTPosCntTexas:
    if num != 0:
        RTPosCntTexas[i] = np.log2(RTPosCntTexas[i])
    i = i+1

i = 0
for num in RTNegCntTexas:
    if num != 0:
        RTNegCntTexas[i] = np.log2(RTNegCntTexas[i])
    i = i+1

width = 0.35       
fig, ax = plt.subplots()
labels = ['0','1-10','10-20','20-50','50-100','100-200','200-500','500-1000','1000-2000','2000-5000','>5000']
ax.bar(labels, RTPosCntTexas, width, label='Pos', color = '#2F329F', alpha = 0.5)
ax.bar(labels, RTNegCntTexas, width, label='Neg', color = '#2F7998', alpha = 0.5, bottom=RTPosCntTexas)
plt.xticks(rotation=45)
ax.set_ylabel('Numero di link nel tweet')
ax.set_ylabel('Frequenza assoluta in scala del logaritmo di 2')
ax.set_title('Sentiment analysis rispetto al numero di retweet\n dei tweet sulle restrizioni in Texas')
ax.legend()
plt.show()

# Pharma

dataPharma = pd.read_excel('C:/Users/gsppp/OneDrive/Poli/Quarto Anno/Business Intelligence per Big Data/Progetto/ProgettoBIClusterAnalysis/Polarity.xlsx',sheet_name=1)
RT_countDFPharma=pd.DataFrame(dataPharma)

RT_countDFPharma['positive'] = [sia.polarity_scores(str(tweet))['pos'] for tweet in RT_countDFPharma.text_clean]
RT_countDFPharma['negative'] = [sia.polarity_scores(str(tweet))['neg'] for tweet in RT_countDFPharma.text_clean]
RT_countDFPharma['neutral'] = [sia.polarity_scores(str(tweet))['neu'] for tweet in RT_countDFPharma.text_clean]
RT_countDFPharma['compound'] = [sia.polarity_scores(str(tweet))['compound'] for tweet in RT_countDFPharma.text_clean]
RT_countDFPharma['overall score'] = [labeler(val) for val in RT_countDFPharma.compound]

RT_countPosDFPharma = RT_countDFPharma.loc[RT_countDFPharma['overall score']=='positive'].sort_values(by='favorite_count')['favorite_count']
RT_countNegDFPharma = RT_countDFPharma.loc[RT_countDFPharma['overall score']=='negative'].sort_values(by='favorite_count')['favorite_count']

# Per chiarezza i conteggi dei "mi piaci" li riportiamo in scala di log2
RTPosCntPharma = RTshrinker(RTCounter(RT_countPosDFPharma))
RTNegCntPharma = RTshrinker(RTCounter(RT_countNegDFPharma))

i = 0
for num in RTPosCntPharma:
    if num != 0:
        RTPosCntPharma[i] = np.log2(RTPosCntPharma[i])
    i = i+1

i = 0
for num in RTNegCntPharma:
    if num != 0:
        RTNegCntPharma[i] = np.log2(RTNegCntPharma[i])
    i = i+1

width = 0.35       
fig, ax = plt.subplots()
labels = ['0','1-10','10-20','20-50','50-100','100-200','200-500','500-1000','1000-2000','2000-5000','>5000']
ax.bar(labels, RTPosCntPharma, width, label='Pos', color = '#2F329F', alpha = 0.5)
ax.bar(labels, RTNegCntPharma, width, label='Neg', color = '#2F7998', alpha = 0.5, bottom=RTPosCntPharma)
plt.xticks(rotation=45)
ax.set_ylabel('Numero di link nel tweet')
ax.set_ylabel('Frequenza assoluta in scala del logaritmo di 2')
ax.set_title('Sentiment analysis rispetto al numero di retweet\n dei tweet sui vaccini')
ax.legend()
plt.show()

# School

dataSchool = pd.read_excel('C:/Users/gsppp/OneDrive/Poli/Quarto Anno/Business Intelligence per Big Data/Progetto/ProgettoBIClusterAnalysis/Polarity.xlsx',sheet_name=2)
RT_countDFSchool=pd.DataFrame(dataSchool)

RT_countDFSchool['positive'] = [sia.polarity_scores(str(tweet))['pos'] for tweet in RT_countDFSchool.text_clean]
RT_countDFSchool['negative'] = [sia.polarity_scores(str(tweet))['neg'] for tweet in RT_countDFSchool.text_clean]
RT_countDFSchool['neutral'] = [sia.polarity_scores(str(tweet))['neu'] for tweet in RT_countDFSchool.text_clean]
RT_countDFSchool['compound'] = [sia.polarity_scores(str(tweet))['compound'] for tweet in RT_countDFSchool.text_clean]
RT_countDFSchool['overall score'] = [labeler(val) for val in RT_countDFSchool.compound]

RT_countPosDFSchool = RT_countDFSchool.loc[RT_countDFSchool['overall score']=='positive'].sort_values(by='favorite_count')['favorite_count']
RT_countNegDFSchool = RT_countDFSchool.loc[RT_countDFSchool['overall score']=='negative'].sort_values(by='favorite_count')['favorite_count']

# Per chiarezza i conteggi dei "mi piaci" li riportiamo in scala di log2
RTPosCntSchool = RTshrinker(RTCounter(RT_countPosDFSchool))
RTNegCntSchool = RTshrinker(RTCounter(RT_countNegDFSchool))

i = 0
for num in RTPosCntSchool:
    if num != 0:
        RTPosCntSchool[i] = np.log2(RTPosCntSchool[i])
    i = i+1

i = 0
for num in RTNegCntSchool:
    if num != 0:
        RTNegCntSchool[i] = np.log2(RTNegCntSchool[i])
    i = i+1

width = 0.35       
fig, ax = plt.subplots()
labels = ['0','1-10','10-20','20-50','50-100','100-200','200-500','500-1000','1000-2000','2000-5000','>5000']
ax.bar(labels, RTPosCntSchool, width, label='Pos', color = '#2F329F', alpha = 0.5)
ax.bar(labels, RTNegCntSchool, width, label='Neg', color = '#2F7998', alpha = 0.5, bottom=RTPosCntSchool)
plt.xticks(rotation=45)
ax.set_ylabel('Numero di link nel tweet')
ax.set_ylabel('Frequenza assoluta in scala del logaritmo di 2')
ax.set_title('Sentiment analysis rispetto al numero di retweet\n dei tweet sulle scuole')
ax.legend()
plt.show()







