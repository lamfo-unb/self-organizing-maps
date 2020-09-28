import pandas as pd
from sklearn.preprocessing import scale
from minisom import MiniSom
from modulos.mapa import Mapa
from modulos.process_data import tratar_dominician_rep

# Lendo dados do índice de democracia
path_data = "./data/processed/democracy-index.csv"
args_csv = {'sep': ';', 'decimal': ',', 'encoding': 'cp1252'}
democracy_index = pd.read_csv(path_data, **args_csv)
democracy_index.dropna(inplace=True)
democracy_index.head()
democracy_index.shape

# Lendo dicionário de siglas para os paises
path = "./data/processed/country-code.csv"
country_code = pd.read_csv(path, **args_csv, na_values='')
country_code.head()

# Transformando nome dos paises para siglas
countries_, codes_ = country_code['country'].to_list(), country_code['code'].to_list()
country_codes = dict(zip(countries_, codes_))

countries = [country_codes[v] for v in democracy_index['country'].to_list()]

# Definindo cores para os países de acordo com o nível de democracia
category_color = {
    'Full democracy': 'darkgreen', 'Flawed democracy': 'limegreen',  'Hybrid regime': 'darkorange',
    'Authoritarian': 'crimson'
}
colors_dict = {c: category_color[dm] for c, dm in zip(democracy_index.country,
                                                      democracy_index.category)}

# Padronizando dados para aplicar o self-organizad map
feature_names = ['democracy_index', 'electoral_processand_pluralism', 'functioning_of_government',
                 'political_participation', 'political_culture', 'civil_liberties']

X = democracy_index[feature_names].values
X = scale(X)

# Treinando SOM
size = 15
som = MiniSom(x=size, y=size, input_len=len(X[0]), sigma=1.5, neighborhood_function='gaussian', random_seed=1)

som.pca_weights_init(X)
som.train_random(X, 1000, verbose=True)

# Plotando mapa
labels_country = [tratar_dominician_rep(x) for x in democracy_index['country']]
country_colors = [v for _, v in colors_dict.items()]
mapa_democracia = Mapa(som, X, labels_country, 15)
mapa_democracia.plot(labels_colors=country_colors, legend_colors=category_color)

