import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
data = pd.read_csv(r'C:\Users\giuli\Documents\GitHub\Progetto_BI2021\data.csv', parse_dates=['created_at'])
df = pd.DataFrame(data)


#DECODING DEL TESTO ALL INTERNO DELLA COLONNA text. 
#all interno della colonna text è presente una rappresentazione di un bytes sotto forma di stringa.
#questa può essere valutata attraverso la funzione ast.literal_eval(string) e quindi decodificata
#secondo la codifica appropriata tramite il metodo decode
text_enc = []
for i in range(0, df.text.size):
    txt= ast.literal_eval(df.text[i]).decode('utf-8')
    text_enc.append(txt)

df['text_enc'] = text_enc



#PROVO A FARE UN PO DI LANGUAGE DETECTION---------------------------------------------------------------------------

from langdetect import detect_langs
import re

#il metodo detect_langs fornisce un vettore di possibilità riguardo alla lingua del testo che sta analizzando
#vado a vedere se ci sono nel dataset testi ambigui (quindi quelli per cui la dimensione del dict ritornato da detect_langs)
#è maggiore di 1

# utilizzo langdetect che è abbastanza veloce per generare una previsione sommaria. Esso restituisce la probabilità per ciascuna lingua
# trovata quindi:
#   - se trovo una sola lingua tra quelle ammesse (en, es, fr) allora segno la lingua trovata e segno la detection come sicura
#   - se trovo più di una lingua oppure trovo come lingua più probabile una non ammessa segno la lingua e segno la detection come insicura/sbagliata 

unsure_count = 0
wrong_count = 0
i=0
lang_detect_before = []
unsure_wrong_before = []
for twt in df.text_enc:
    try:
        detection = detect_langs(twt)
        lang = detection[0].lang
        
        lang_detect_before.append(lang)
        unsure_wrong_before.append(False)
        
        if((lang != "en") and (lang != "es") and (lang != "fr")):
             wrong_count = wrong_count +1
             unsure_wrong_before[i]=True
        if(len(detection)>1):
            unsure_wrong_before[i]=True
            unsure_count = unsure_count+1

    except:
        print("errore alla posizione ")
        i
        print("testo : " + twt)
        lang_detect_before.append('None')
        unsure_wrong_before.append(True)

    i=i+1



df['langdetect']=lang_detect
df['unsure_wrong_detection'] = unsure_wrong

unsure_wrong_before_sum = df['unsure_wrong_detection'].sum()
unsure_wrong_before_sum

#PREVENTIVO AL LANGUAGE DETECTION: PULIZIA DEI TWEET
# vado a vedere se la pulizia dei tweet migliora le performance della traduzione

def cleanTxt(text):
 text = re.sub('@[A-Za-z0–9]+', '', text) #Rimuove le @menzioni
 text = re.sub('#', '', text) # Rimuove l'hashtag
 text = re.sub('https?:\/\/\S+', '', text) # Rimuove i link
 return text


df['text_clean'] = df.text_enc.apply(cleanTxt)

#--------performo di nuovo la traduzione
unsure_count = 0
wrong_count = 0
i=0
lang_detect = []
unsure_wrong = []
for twt in df.text_clean:
    try:
        detection = detect_langs(twt)
        lang = detection[0].lang
        
        lang_detect.append(lang)
        unsure_wrong.append(False)
        
        if((lang != "en") and (lang != "es") and (lang != "fr")):
             wrong_count = wrong_count +1
             unsure_wrong[i]=True
        if(len(detection)>1):
            unsure_wrong[i]=True
            unsure_count = unsure_count+1
    except:
        print("errore alla posizione ")
        i
        print("testo : " + twt)
        lang_detect.append('None')
        unsure_wrong.append(True)

    i=i+1


df['langdetect']=lang_detect
df['unsure_wrong_detection'] = unsure_wrong

unsure_wrong_after_sum = df['unsure_wrong_detection'].sum()
unsure_wrong_after_sum

#pulire il testo migliora di pochissimo (quasi niente) le performance della language detection

#CHECK DELLA TRADUZIONE CON API GOOGLE
#Questa API non è stata utilizzata prima perché estremamente lenta e non permette di effettuare troppe richieste ai server google

from textblob import TextBlob

lang_detect_final = []
i = 0
for twt in df.text_clean:
    if not df['unsure_wrong_detection'][i]:
        lang_detect_final.append(df['langdetect'][i])
    else:
        blob = TextBlob(twt)
        lang = blob.detect_language()
        if((lang != "en") and (lang != "es") and (lang != "fr")):
            lang_detect_final.append('None')
            print("nesun match alla riga")
            i
        else:
            lang_detect_final.append(lang)
    i=i+1

df['lang_detect_final'] = lang_detect_final



#ESPORTO I DATI NUOVI IN EXCEL PER ANALIZZARLI SU RAPID MINER
#excel non supporta le date con le timezone quindi le ho dovute eliminare con il metodo datetime.tz_localize()
df_enc = df[['favorite_count', 'source', 'text_enc', 'is_retweet', 'retweet_count', 'created_at']]
df_enc.loc[:,'created_at_ntz']= df_enc.created_at.dt.tz_localize(None)
df_enc.drop(columns='created_at')
df_enc = df_enc.drop(columns='created_at')
df_enc.to_excel('data_dec.xlsx')
