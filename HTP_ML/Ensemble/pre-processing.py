from utility import magpie_it
import pandas as pd
from matminer.featurizers.composition import ElementProperty

elementproperty = ElementProperty.from_preset("magpie")

df = pd.read_csv('magHEA_quat.csv')
print(df.describe())
# Element1,Element2,Element3,Element4,afm,lat,sws0,sws_init,mag,B0,TC
columns = ['Formula','afm','lat','sws0','B0','mag','TC']
# columns = ['Element1', 'Element2', 'Element3', 'Element4', 'Element5', 'afm','lat','sws0','B0','mag','TC']
if 'Element5' in df.columns:
    columns.append('Element5')
    df['Formula'] = df[['Element1', 'Element2', 'Element3', 'Element4', 'Element5']].agg('0.20'.join, axis=1) + '0.20'
else:
    df['Formula'] = df[['Element1', 'Element2', 'Element3', 'Element4']].agg('0.25'.join, axis=1) + '0.25'
data = df[columns]

print(data.describe())
print(data.head())

samples = magpie_it(data["Formula"])
labels = elementproperty.feature_labels()
df_features = pd.DataFrame(samples, columns=labels)
# df = pd.DataFrame(data, columns=column_labels)
print(df_features.describe())

df = pd.concat([data, df_features], axis=1)
print(df)
# print(df[['B0', 'MagpieData mean BulkModulus']].describe())
#print(df[['G', 'MagpieData mean ShearModulus']].describe())
df.to_csv('data.csv', index=False)
