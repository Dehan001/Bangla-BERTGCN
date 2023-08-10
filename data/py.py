import pandas as pd
from sklearn.model_selection import train_test_split
#df_train=pd.read_csv("Train.csv")

#df_test=pd.read_csv("Test.csv")

#df_val= pd.read_csv('Val.csv')

# list=[df_train,df_test,df_val]

# df = pd.concat([df_train.assign(ind="train"),df_test.assign(ind="test")])
# try:
#     df = pd.read_csv('Emotion.csv', error_bad_lines=False, sep=';')
    
# except pd.errors.ParserError as e:
#     print("Error occurred while parsing the CSV file:", e)


# df=df.loc[df['lan'] == 'BN']
# df.reset_index(drop=True, inplace=True)
# df.drop(['id','domain','lan'], axis=1)
# print(df.head())
df=pd.read_csv('/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/SarcasDetectionEDA.csv')
# train,test = train_test_split(df, test_size=0.33, random_state=42,shuffle=False)

# df = pd.concat([train.assign(ind="train"),test.assign(ind="test")])
# df_corpus=df[['ind','Label']]
# df_corpus.reset_index(inplace = True)
# df_corpus.to_csv('SarcasDetection(1).txt', sep='\t', index=False, header=False)

# df['Comments'].to_csv('SentNOB.txt', sep='\t')
# df_corpus=df["Comments"]
# df['Comments'].to_csv('SarcasDetection.txt', sep='\t', index=False, header=False)

# df_corpus=df[['ind','Label']]
# df['text'].to_csv('Emotion.txt', sep='\t')
# df_corpus=df["Data"]
# df['Data'].to_csv('SentNOB(1).txt', sep='\t', index=False, header=False)


from normalizer import normalize
df['Comments'] = df['Comments'].apply(normalize)
df['Comments'].to_csv('SarcasDetection.clean.txt', sep='\t', index=False, header=False)
