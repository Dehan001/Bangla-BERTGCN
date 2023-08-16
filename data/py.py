import pandas as pd
from sklearn.model_selection import train_test_split
# train=pd.read_csv("/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/train.csv")

# test=pd.read_csv("/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/test.csv")
# df=pd.read_csv('/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/finaldataset.csv')
#df_val= pd.read_csv('Val.csv')
auth=pd.read_csv('/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/LabeledAuthentic-7K.csv')
fake=pd.read_csv('/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/LabeledFake-1K.csv')

df = auth[:700]
df = df.append(fake[:100])
# datapath='/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/Sentiment and emotion/archive(6)/Sentiment.csv'

# try:
#     df = pd.read_csv(datapath, on_bad_lines='skip', sep=';')

# except pd.errors.ParserError as e:
#     print("Error occurred while parsing the CSV file:", e)



df = df.sample(frac=1).reset_index(drop=True)
# df=df.loc[df['lan'] == 'BN']

# df['content']=df['content'][:100]
from sklearn.model_selection import train_test_split

train, test= train_test_split(df, test_size=0.33, random_state=121, stratify=df['label'])


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

# df.reset_index(drop=True, inplace=True)
# df.drop(['id','domain','lan','score'], axis=1)
# print(df.head())
# df=pd.read_csv('/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/Sentiment.csv')
# train,test = train_test_split(df, test_size=0.33, random_state=121,shuffle=True,stratify=df['tag'])

# # print(df.head())
df1 = pd.concat([train.assign(ind="train"),test.assign(ind="test")])

df1.reset_index(inplace=True)
# print(df.head())
df_corpus=df1[['ind','label']]
# df_corpus.reset_index()
df_corpus.to_csv('BanFake(1).txt', sep='\t', index=True, header=False)

# df['Comments'].to_csv('SentNOB.txt', sep='\t')
# df_corpus=df["Comments"]
# df['headline'].to_csv('BanFake.txt', sep='\t', index=False, header=False)

# df_corpus=df[['ind','Label']]
# df['text'].to_csv('Emotion.txt', sep='\t')
# df_corpus=df["Data"]
df['content'].to_csv('BanFake.txt', sep='\t', index=False, header=False)


from normalizer import normalize
df['content'] = df['content'].apply(normalize)
df['content'].to_csv('BanFake.clean.txt', sep='\t', index=False, header=False)
