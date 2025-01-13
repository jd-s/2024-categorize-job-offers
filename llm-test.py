from ollama import chat
from ollama import ChatResponse
import sqlite3
import pandas as pd
import numpy as np

modelname = "command-r"

filename = "output-"+modelname+".csv"

template1 = """Please assign the following tweet to the following categories:

1 Tweets unrelated to the profession

2 tweets with information about the profession

3 tweets with information about a person with this profession

4 tweets with information on education, not the profession

5 Tweets mit Stellenanzeigen

Here is the tweet:

"XYZ"

Please print only one category number and select only one. Print only the number, no text."""

template2 = """Please assign the following tweet to the following categories:

1 Tweets unrelated to the profession

Example: "Aus der Dlf Audiothek | Deep Talk | Sektionsassistentin Belz | Louisa Belz, wie ist das, ein Hirn in der Hand zu halten?
https://t.co/TIIYQMwEdl"

2 tweets with information about the profession

Example: "@recht_nett Fachinformatiker für Systemintegration Azubi im TVL (Tarifvertrag der Länder), in Sachsen im 3. Lehrjahr knapp 1190€ brutto. Netto sind das knapp 900€. Arbeitszeiten sind 40h/Woche. Allerdings auch 13. Gehalt."

3 tweets with information about a person with this profession

Example: "@epsilon3141 Der Mathematiker Weitz zeigt sehr unterhaltsam die  logischen Unzulänglichkeiten von ChatGPT auf und kommt zum Schluss, dass das LLM eher in vermeintlich  kreativen Berufen Jobs vernichten wird.
https://t.co/gREi5fgBJa"

4 tweets with information on education, not the profession

Example: "@ResiReynolds @Lunaminki @Martin_Dete Habe es bei einer Bekannten mitbekommen. Hat einen Ausbilderschein gemacht, die Azubis des Betriebes betreut, ein Fernstudium zur Fachwirtin, neben der Arbeit absolviert. Wer hat die Beförderung bekommen? Der männliche Kollege, der das nicht hatte."

5 Tweets mit Stellenanzeigen

Example: "Ausbildung für 2023 gesucht? Dann hier entlang: Parker Hannifin in #BietigheimBissingen sucht Azubis als #Industriekauffrau #Fachinformatiker oder #FachkraftfürLagerlogistik https://t.co/exvTg8VPns #ausbildungzukunftgestalten #job https://t.co/oyr9I7U6O7"

Here is the tweet:

"XYZ"

Please print only one category number and select only one. Print only the number, no text."""

template3 = """Please assign the following tweet to the following categories:

1 Tweets unrelated to the profession

Example: "Aus der Dlf Audiothek | Deep Talk | Sektionsassistentin Belz | Louisa Belz, wie ist das, ein Hirn in der Hand zu halten?
https://t.co/TIIYQMwEdl"

Another example: "@Valentinafnbr Wie ist fachinformatiker bis jetzt so?"

2 tweets with information about the profession

Example: "@recht_nett Fachinformatiker für Systemintegration Azubi im TVL (Tarifvertrag der Länder), in Sachsen im 3. Lehrjahr knapp 1190€ brutto. Netto sind das knapp 900€. Arbeitszeiten sind 40h/Woche. Allerdings auch 13. Gehalt."

Another example: "Als Azubi in luftigen Höhen. Kaminfegerin auf Dächern unterwegs und ständig in Bewegung. #Schornsteinfeger #Abtsgmünd http://bit.ly/brk5EK"


3 tweets with information about a person with this profession

Example: "@epsilon3141 Der Mathematiker Weitz zeigt sehr unterhaltsam die  logischen Unzulänglichkeiten von ChatGPT auf und kommt zum Schluss, dass das LLM eher in vermeintlich  kreativen Berufen Jobs vernichten wird.
https://t.co/gREi5fgBJa"

Another example: "Französische Genforscherin und Nobelpreisträgerin Emmanuelle Marie Charpentier an Päpstliche Akademie der Wissenschaften berufen.
https://t.co/qlzFbLQPiR"


4 tweets with information on education, not the profession

Example: "@ResiReynolds @Lunaminki @Martin_Dete Habe es bei einer Bekannten mitbekommen. Hat einen Ausbilderschein gemacht, die Azubis des Betriebes betreut, ein Fernstudium zur Fachwirtin, neben der Arbeit absolviert. Wer hat die Beförderung bekommen? Der männliche Kollege, der das nicht hatte."

Another example: "Die ersten 1 1/2 Jahre der #Ausbildung zum #Fachinformatiker sind gerockt.
Gestern, am 01. März 2023, durften alle Azubis im 2. Lehrjahr mit vielen weiteren, den 1. Teil ihrer #Abschlussprüfung schreiben.
Dank sehr engagierter Leute bei der #Lecos waren wir gut vorbereitet"


5 Tweets mit Stellenanzeigen

Example: "Ausbildung für 2023 gesucht? Dann hier entlang: Parker Hannifin in #BietigheimBissingen sucht Azubis als #Industriekauffrau #Fachinformatiker oder #FachkraftfürLagerlogistik https://t.co/exvTg8VPns #ausbildungzukunftgestalten #job https://t.co/oyr9I7U6O7"

Another example: "RT @PolizeiMuenchen: #Stellenangebot
Wir suchen für den Standort #München zwei Azubis zum Fachinformatiker in der Fachrichtung Systemintegr…"

Another example: "Stellenangebot: Karriere als Offizier und Arzt (m/w) im Sanitätsdienst der Bundeswehr in deutschlandweit https://t.co/QFpwe1LwsD"

Another example: "#jobs #jobsearch #Berlin #KARRIERE ALS OFFIZIER IM TRUPPENDIENST (M/W):  
           #bundesweit, Karriere al... https://t.co/CSsLL2bmtB"


Here is the tweet:

"XYZ"

Please print only one category number and select only one. Print only the number, no text."""

conn = sqlite3.connect('/home/jens/database2.db')
qu = """SELECT * FROM Tweets where Tweets.job_id LIKE'%B 4%' and Tweets.category != 0"""#"""+str(kat[0])+"""%" """
df_train = pd.read_sql(qu, conn)

#df_train = df_train.groupby('category').sample(df_train.groupby('category').size().min())
from sklearn.utils import resample

minority_class = df_train[df_train['category'] <= 4]
majority_class = df_train[df_train['category'] == 5]

# Downsample the majority class
majority_downsampled = resample(majority_class, replace=False, n_samples=27, random_state=42)

# Combine the downsampled majority class with the minority class
df_train = pd.concat([minority_class, majority_downsampled ])


for index, row in df_train.iterrows():
    #print(row['tweet_txt'], row['category'])
    response: ChatResponse = chat(model=modelname, messages=[
          {
            'role': 'user',
            'content': template1.replace("XYZ", row['tweet_txt'])
          },
        ])
    r1 = response['message']['content'].strip()
    #rint ("===\n"+r1)
    response: ChatResponse = chat(model=modelname, messages=[
          {
            'role': 'user',
            'content': template2.replace("XYZ", row['tweet_txt'])
          },
        ])
    r2 = response['message']['content'].strip()
    #rint ("===\n"+r2)
    response: ChatResponse = chat(model=modelname, messages=[
          {
            'role': 'user',
            'content': template3.replace("XYZ", row['tweet_txt'])
          },
        ])
    r3 = response['message']['content'].strip()
    #print ("===\n"+r3)
    #print (str(row['category']).strip()+";"+r1+";"+r2+";"+r3+"\n")
    with open(filename, 'a') as fd:
        fd.write(str(row['category']).strip()+";"+r1+";"+r2+";"+r3+"\n" )
