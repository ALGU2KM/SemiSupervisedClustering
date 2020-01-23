import pandas as pd

dados = pd.read_csv('d:/basedados/mapa.csv')

colunas = ['B1', 'B10', 'B11', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7','B8', 'B9', 'BQA', 'EVI', 'NDVI', 'landcover', 'slope']

mapa = dados[colunas]
mapa.to_csv('d:/basedados/matupiba2.csv', index=False)

