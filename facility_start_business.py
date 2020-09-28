
import pandas as pd
from sklearn.preprocessing import scale
from minisom import MiniSom

from modulos.mapa import Mapa


# Carregando dados
path_data = 'data/processed/start-business.csv'
read_params = {'sep': ';', 'decimal': ',', 'encoding': 'cp1252'}
dados = pd.read_csv(path_data, **read_params)

# Preparando dados
features = ['procedures_man', 'time_men', 'cost_men', 'procedures_woman', 'time_woman', 'cost_woman','paid_in_min']
labels_contry = dados['country_code']
X = dados[features].values
X = scale(X)

# Treinando SOM
size = 15
som = MiniSom(x=size, y=size, input_len=len(X[0]), sigma=1.5, neighborhood_function='gaussian', random_seed=1)
som.pca_weights_init(X)
som.train_random(X, 1000, verbose=True)

# Instanciando classe para gráficos
mapa = Mapa(som, X, labels_contry, size)

# Mapa dos países relacionado-o com grupos por facilidade de abrir um novo negócio
facility_colors = {
    'Low facility': '#DC143C', 'Lower middle facility': '#FFA500', 'Upper middle facility': '#9370DB',
    'High facility': '#4B0082'
}
country_start_colors = [facility_colors[facility] for facility in dados['facility_group'].to_list()]
mapa.plot(country_start_colors, facility_colors)

# Mapa dos países relacionado-o com grupos por renda
income_colors = {
    'Low income': '#DC143C', 'Lower middle income': '#FFA500', 'Upper middle income': '#9370DB', 'High income': '#4B0082'
}
country_income_colors = [income_colors[income] for income in dados['income_group'].to_list()]
mapa.plot(labels_colors=country_income_colors, legend_colors=income_colors)
