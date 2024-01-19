import pandas as pd
from sklearn.model_selection import train_test_split
# train=pd.read_csv("/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/train.csv")

# test=pd.read_csv("/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/test.csv")
# df=pd.read_csv('/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/finaldataset.csv')
#df_val= pd.read_csv('Val.csv')
# auth=pd.read_csv('/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/LabeledAuthentic-7K.csv')
# fake=pd.read_csv('/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/LabeledFake-1K.csv')
df=pd.read_csv("C:\Users\ndc\Documents\Thesis\experiment5-2\data\BanglaError.csv")
# df = auth
# df = df.append(fake)
# df['headline']=df['headline']+" [SEP] "+df['content']
# df['headline_words'] = df['headline'].str.split()

# Rejoining the first 128 words
# df['text'] = df['text'].apply(lambda words: ' '.join(words[:128]))
# df=df.head(3000)
# def extract_first_128_words(text):
#     words = text.split()[:128]
#     return ' '.join(words)

# Apply the function to the 'headline' column
# df['headline'] = df['headline'].apply(extract_first_128_words)

# datapath='/home/farhan/Documents/nlp/bertgcn-bangla/BERTGCN/data/Sentiment and emotion/archive(6)/Sentiment.csv'

# try:
#     df = pd.read_csv(datapath, on_bad_lines='skip', sep=';')

# except pd.errors.ParserError as e:
#     print("Error occurred while parsing the CSV file:", e)



# df = df.sample(frac=1).reset_index(drop=True)
# df=df.loc[df['lan'] == 'BN']

# df['content']=df['content'][:100]
from sklearn.model_selection import train_test_split

train, test= train_test_split(df, test_size=0.25, random_state=121, stratify=df['Error'])


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
df_corpus=df1[['ind','Error']]
df_corpus1=df1[['ind','Category']]
# df_corpus.reset_index()
df_corpus.to_csv('BanglaErrorBin.txt', sep='\t', index=True, header=False)
df_corpus1.to_csv('BanglaErrorMulti.txt', sep='\t', index=True, header=False)
# df['Comments'].to_csv('SentNOB.txt', sep='\t')
# df_corpus=df["Comments"]
# df['headline'].to_csv('BanFake.txt', sep='\t', index=False, header=False)

# df_corpus=df[['ind','Label']]
# df['text'].to_csv('Emotion.txt', sep='\t')
# df_corpus=df["Data"]
# df['Comment'].to_csv('BanglaErrorBin(1).txt', sep='\t', index=False, header=False)
# df['Comment'].to_csv('BanglaErrorMulti(1).txt', sep='\t', index=False, header=False)

# # from normalizer import normalize
# # df['Comment'] = df['Comment'].apply(normalize)
# df['Comment'].to_csv('BanglaErrorBin(1).clean.txt', sep='\t', index=False, header=False)
# df['Comment'].to_csv('BanglaErrorMulti(1).clean.txt', sep='\t', index=False, header=False)
