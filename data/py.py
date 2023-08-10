import pandas as pd
df_train=pd.read_csv("Train.csv")

df_test=pd.read_csv("Test.csv")

df_val= pd.read_csv('Val.csv')

list=[df_train,df_test,df_val]

df = pd.concat([df_train.assign(ind="train"),df_test.assign(ind="test")])


#df_corpus=df[['ind','Label']]
#df['Data'].to_csv('SentNOB.txt', sep='\t')
#df_corpus=df["Data"]
#df['Data'].to_csv('SentNOB(1).txt', sep='\t', index=False, header=False)

from normalizer import normalize
df['Data'] = df['Data'].apply(normalize)
df['Data'].to_csv('SentNOB.clean.txt', sep='\t', index=False, header=False)
