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

#PROVO A FARE UN PO DI LANGUAGE DETECTION
from langdetect import detect_langs
#il metodo detect_langs fornisce un vettore di possibilità riguardo alla lingua del testo che sta analizzando
#vado a vedere se ci sono nel dataset testi ambigui (quindi quelli per cui la dimensione del dict ritornato da detect_langs)
#è maggiore di 1
for twt in df.text_enc:
    detection = detect_langs(twt)
    if(len(detection)>1):
        print(twt)
        print(detection)


#ESPORTO I DATI NUOVI IN EXCEL PER ANALIZZARLI SU RAPID MINER
#excel non supporta le date con le timezone quindi le ho dovute eliminare con il metodo datetime.tz_localize()
df_enc = df[['favorite_count', 'source', 'text_enc', 'is_retweet', 'retweet_count', 'created_at']]
df_enc.loc[:,'created_at_ntz']= df_enc.created_at.dt.tz_localize(None)
df_enc.drop(columns='created_at')
df_enc = df_enc.drop(columns='created_at')
df_enc.to_excel('data_dec.xlsx')
