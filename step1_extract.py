import pandas as pd
import numpy as np

df = pd.read_csv('ADNIMERGE_20220902.csv', low_memory=False)  #data input
df2=pd.read_csv('unfind_imageid.csv')  #Read missing list of missing imageID
df = df[~df['IMAGEUID'].isin(df2['imageid'].tolist())]    
print(df)
df = df.to_csv('ADNIMERAGE_data.csv', index=False)  #output

df = pd.read_csv('ADNIMERAGE_data.csv', low_memory=False)  #data input
df['IMAGEUID'] = df['IMAGEUID'].fillna(-1)
df = df[df['IMAGEUID'] != -1]   # Pick out the data whose image is not a null value
cnt = df['RID'].value_counts()  #Index by RID
idx = cnt[cnt > 1].index.tolist()  #idx>1
df = df[(df['RID'].isin(idx))]  #Indexing with idx
df = df.sort_values(by=['RID', 'Month'])
df = df.to_csv('step1_merge_data_ADNI.csv', index=False)
print()




