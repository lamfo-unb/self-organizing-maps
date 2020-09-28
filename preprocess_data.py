import pandas as pd
from modulos.process_data import niveis_facilidade

# Importando dados
path_data = "data/raw/start-business.xlsx"
dados = pd.read_excel(path_data)
dados.head()

# Selecionando colunas
colunas = pd.Series(dados.columns)
colunas = colunas[~colunas.str.contains("Score")].to_list()
colunas.append("Score-Starting a business")
dados = dados[colunas]
dados.drop(columns=['DB Year'], inplace=True)
dados.head()

# Renomeando colunas
new_columns = [
    'country_code', 'economy', 'region', 'income_group', 'procedures_man', 'time_men', 'cost_men', 'procedures_woman',
    'time_woman', 'cost_woman', 'paid_in_min', 'score'
]
dados.columns = new_columns
dados.head()

# Retirando pais com diferencial para cidade
dados = dados.query("~country_code.str.contains('_')")

# Definido grupos paises por nível de facilidade de abrir negócios
dados['facility_group'] = niveis_facilidade(dados['score'])

# Ordenando colunas
new_order_columns = [
    'country_code', 'economy', 'region', 'income_group', 'facility_group', 'score', 'procedures_man',
    'time_men', 'cost_men', 'procedures_woman', 'time_woman', 'cost_woman', 'paid_in_min'
]
dados = dados[new_order_columns]

# Exportando resultados
path_out = 'data/processed/start-business.csv'
write_params = {'sep': ';', 'decimal': ',', 'index': False, 'encoding': 'cp1252'}
dados.to_csv(path_out, **write_params)
