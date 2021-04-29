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
df

