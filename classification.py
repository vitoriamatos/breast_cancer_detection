import pandas as pd
from pycaret.classification import *
 
# Suponha que você tenha um DataFrame chamado df
# df = pd.read_csv('seu_arquivo.csv')
 
# Configuração inicial do PyCaret
clf1 = setup(data=df, target='chavedesaida', silent=True, session_id=123)
 
# Treinando todos os modelos
best_model = compare_models()
 
print(best_model)



# Depois de executar o setup
clf1 = setup(data=df, target='chavedesaida', silent=True, session_id=123)
 
# Treinando e comparando os modelos, e capturando os resultados em um DataFrame
results = pull()  # Esta função captura a última saída exibida
print(results)
