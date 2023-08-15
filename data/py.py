import pandas as pd
from sklearn.model_selection import train_test_split
train=pd.read_csv("/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/train.csv")

test=pd.read_csv("/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/test.csv")
# df=pd.read_csv('/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/finaldataset.csv')
#df_val= pd.read_csv('Val.csv')

# list=[df_train,df_test,df_val]
# list=[df_train,df_test]
# df = pd.concat([df_train,df_test])
# df = df.sample(frac = 1)
# df.reset_index(drop=True, inplace=True)

# try:
#     df = pd.read_csv('/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/emotion', on_bad_lines='skip', sep='\t')
    
# except pd.errors.ParserError as e:
#     print("Error occurred while parsing the CSV file:", e)

# print(df.head())

# df=df.loc[df['lan'] == 'BN']
# df.reset_index(drop=True, inplace=True)
# df.drop(['id','domain','lan','score'], axis=1)
# print(df.head())
# df=pd.read_csv('/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/Sentiment.csv')
# train,test = train_test_split(df, test_size=0.33, random_state=121,shuffle=True,stratify=df['tag'])

# # print(df.head())
df = pd.concat([train.assign(ind="train"),test.assign(ind="test")])

df.reset_index(inplace=True)
# print(df.head())
df_corpus=df[['ind','label']]
# df_corpus.reset_index()
df_corpus.to_csv('BengaliHateSpeech(1).txt', sep='\t', index=True, header=False)

# df['Comments'].to_csv('SentNOB.txt', sep='\t')
# df_corpus=df["Comments"]
# df['headline'].to_csv('BanFake.txt', sep='\t', index=False, header=False)

# df_corpus=df[['ind','Label']]
# df['text'].to_csv('Emotion.txt', sep='\t')
# df_corpus=df["Data"]
df['text'].to_csv('BengaliHateSpeech.txt', sep='\t', index=False, header=False)


from normalizer import normalize
df['text'] = df['text'].apply(normalize)
df['text'].to_csv('BengaliHateSpeech.clean.txt', sep='\t', index=False, header=False)
