import pandas as pd
from sklearn.model_selection import train_test_split
#df_train=pd.read_csv("Train.csv")

#df_test=pd.read_csv("Test.csv")

#df_val= pd.read_csv('Val.csv')

# list=[df_train,df_test,df_val]

# df = pd.concat([df_train.assign(ind="train"),df_test.assign(ind="test")])
try:
    df = pd.read_csv('Emotion.csv', error_bad_lines=False, sep=';')
    
except pd.errors.ParserError as e:
    print("Error occurred while parsing the CSV file:", e)


df=df.loc[df['lan'] == 'BN']
df.reset_index(drop=True, inplace=True)
df.drop(['id','domain','lan'], axis=1)
print(df.head())

train,test = train_test_split(df, test_size=0.33, random_state=42,shuffle=False)

df = pd.concat([train.assign(ind="train"),test.assign(ind="test")])
df_corpus=df[['ind','emotion']]
df_corpus.reset_index(inplace = True)
df_corpus.to_csv('Emotion(1).txt', sep='\t', index=False, header=False)

#df['Data'].to_csv('SentNOB.txt', sep='\t')
#df_corpus=df["Data"]
#df['Data'].to_csv('SentNOB(1).txt', sep='\t', index=False, header=False)

# df_corpus=df[['ind','Label']]
# df['text'].to_csv('Emotion.txt', sep='\t')
# df_corpus=df["Data"]
# df['Data'].to_csv('SentNOB(1).txt', sep='\t', index=False, header=False)


# from normalizer import normalize
# df['Data'] = df['Data'].apply(normalize)
# df['Data'].to_csv('SentNOB.clean.txt', sep='\t', index=False, header=False)
