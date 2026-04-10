import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=[8.0, 10.0, 13.0, 15.0], y=[8.04, 6.95, 7.58, 8.81], 
                    line=dict(color='red', width=2, dash="solid", shape='hv'), 
                    mode="lines", name='Line chart'))
fig1.show()


# Descarregar e analisar o datset
df = pd.read_csv('olympics.csv')
print(f"(linhas, colunas) = {df.shape}","\n")
print(df.info(), "\n")
print(df.head(),"\n")
for i, col in enumerate(df.columns):
    if i in [3, 4, 5, 9]: # Colunas numéricas
        print(f"Número de valores diferentes em {col}: {df[col].nunique()}, (desde {df[col].min()} a {df[col].max()})")
    else:
        print(f"Número de valores diferentes em {col}: {df[col].nunique()}") # nº de valores diferentes na coluna

# Valores nulos
missing_values = df.isnull().sum()
print("\nValores nulls por coluna:")
print(missing_values, "\n")

# Retirar as pessoas que não receberam medalha
df = df.dropna(subset=['Medal'])

# linhas duplicadas
duplicate_rows = df.duplicated().sum()
df = df.drop_duplicates()
print(f'Linhas duplicadas: {duplicate_rows}', "\n")

# Matriz de correlações
numeric_df = df.select_dtypes(include=['float64', 'int64'])  # selecionar só as variáveis numéricas
corr = numeric_df.corr()
fig = go.Figure(data=go.Heatmap(z=corr.values, # valores das correlações
                                x=corr.columns, 
                                y=corr.columns, 
                                colorscale="RdBu", # red: -1, blue: 1
                                zmin=-1, zmax=1))  # correlações variam entre -1 e 1
fig.update_layout(title="Mapa de Correlações entre Atributos")
fig.show()

# Pairplot
cols_to_use = ["Age", "Height", "Season", "Weight", "Year", "Sex"]  # experimetar com várias variáveis
sns.pairplot(df[cols_to_use], hue = "Sex", diag_kind = "kde", palette = {"M": "blue", "F": "red"}) # mostra a densidade na diagonal
plt.show()

# Selecionar os três atributos 
cols = ['Sex', 'Year', 'Season']
df = df[cols]

# Descobrir em que anos houve duas estações com jogos
season_count_per_year = df.groupby('Year')['Season'].nunique()  # matriz com nº de seasons de cada ano
for ano, n_estacoes in season_count_per_year.items():
    print(f"Ano {ano}: {n_estacoes} {'estação' if n_estacoes == 1 else 'estações'} com provas")
    
# Lista dos anos que têm duas seasons de provas
anos_validos = [1924, 1928, 1932, 1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992]

# Filtrar apenas as linhas correspondentes a esses anos
df = df[df['Year'].isin(anos_validos)]

# Renomear as colunas para algo mais legível
df['Sex'] = df['Sex'].replace({'M': 'Homens', 'F': 'Mulheres'})

# Conta quantos registos há por ano, estação e sexo (coloca 0 onde não há)
final_df = df.groupby(['Year', 'Season', 'Sex']).size().unstack(fill_value=0)

print("\nDATASET FINAL:")
print(final_df)
print(df.columns)



# Separar as amostras por estação (verão e inverno), que ainda não corresponde ao Datset Final
summer = df[df['Season'] == 'Summer']
winter = df[df['Season'] == 'Winter']
print(summer)
print(winter)

# Agrupar por ano e sexo para reconstruir o dataset final (sem alterar os dados)
summer_counts = summer.groupby(['Year','Sex']).size().unstack(fill_value=0) # Dados do verão
winter_counts = winter.groupby(['Year','Sex']).size().unstack(fill_value=0) # Dados do inverno
print(summer_counts)
print(winter_counts)

fig1 = go.Figure()

# ---- Verão ----
fig1.add_trace(go.Scatter(
    x = summer_counts.index, 
    y = summer_counts['Homens'], 
    mode = 'lines+markers',
    name = 'Verão (Homens)',
    line = dict(color = '#2196f3'),
    marker = dict(size = 10, symbol = 'circle')
))
fig1.add_trace(go.Scatter(
    x = summer_counts.index,
    y = summer_counts['Mulheres'],
    mode = 'lines+markers',
    name = 'Verão (Mulheres)',
    line = dict(color = '#f74780'),
    marker = dict(size = 10, symbol = 'circle')
))

# ---- Inverno ----
fig1.add_trace(go.Scatter(
    x = winter_counts.index,
    y = winter_counts['Homens'],
    mode = 'lines+markers',
    name = 'Inverno (Homens)',
    line = dict(color = '#c3e3fd'),
    marker = dict(size = 10, symbol = 'square')
))
fig1.add_trace(go.Scatter(
    x = winter_counts.index,
    y = winter_counts['Mulheres'],
    mode = 'lines+markers',
    name = 'Inverno (Mulheres)',
    line = dict(color = '#ffc1d5'),
    marker = dict(size = 10, symbol = 'square')
))

fig1.update_layout(
    title = dict(
        text = '<b>Evolução dos medalhistas por sexo e estação ao longo dos anos</b>',
        font = dict(color = '#1E88E5', size = 27, family = "Calibri"),
        x = 0.4 
    ),
    xaxis = dict(
        title = dict(text = 'Ano', font = dict(size = 16, color = "#020202")),
        gridcolor = '#FFB5C0',
        tickvals = summer_counts.index, # aparecer só os anos que houve provas (eixo x)
        tickangle = -40 # valores do eixo x inclinados
    ),
    yaxis = dict(
        title = dict(text = 'Número de Atletas', font = dict(size = 16, color = "#020202")),
        gridcolor = '#FFB5C0',
        rangemode = 'tozero' # eixo do y começa em zero
    ),
    legend_title = dict(text = 'Categorias:', font = dict(size = 14)),
    plot_bgcolor = "#FFF9FA"  # cor de fundo do gráfico
)

fig1.show()



from plotly.subplots import make_subplots
import math

# Lista de anos disponíveis no dataset de inverno
anos = winter_counts.index.tolist()

# Número de colunas e linhas dos subplots
n_cols = 8
n_rows = math.ceil(len(anos) / n_cols)

# Criar subplots 
fig2 = make_subplots(
    rows = n_rows, 
    cols = n_cols,
    # define cada célula como um gráfico circular (domain)
    specs = [[{'type':'domain'}] * n_cols for i in range(n_rows)],  
    # título de cada subplot é o ano
    subplot_titles=[f"<span style='font-weight:bold; color:'rgb(8,48,107)'; font-size:15.5px;'>{ano}</span>" for ano in anos] 
)

# Adicionar um gráfico Pie para cada ano
for i, ano in enumerate(anos):
    
    fig2.add_trace(
        go.Pie(
            labels = ['Verão (Homens)', 'Verão (Mulheres)', 'Inverno (Homens)', 'Inverno (Mulheres)'],
            values = [
                summer_counts.loc[ano, 'Homens'],
                summer_counts.loc[ano, 'Mulheres'],
                winter_counts.loc[ano, 'Homens'],
                winter_counts.loc[ano, 'Mulheres']
            ],
            marker_colors = ['#2196f3', '#f74780', '#c3e3fd', '#ffc1d5'],  
            hole = 0.0,   # tamanho do buraco central 
            showlegend = (i==0),  # mostrar legenda apenas uma vez para o primeiro Pie
            textinfo = 'percent',  # mostrar a percentagem
            insidetextfont = dict(size = 12), # tamanho da percentagem
            textposition = 'inside',  # percentagem dentro da fatia
            sort = False,    # manter sempre a mesma ordem para vizualizar melhor
            direction = 'clockwise',  # sentido dos ponteiros
            rotation = 0,   # a primeira fatia começa sempre no mesmo angulo do círculo
            pull = [0.2, 0, 0, 0], # destacar a categoria "Verão (homens)" para visualizar melhor
            scalegroup = 'one' # pies maiores para anos com maior nº total de medalhistas
        ),
        row =  i // n_cols + 1, # divisão inteira
        col = i % n_cols + 1    # resto da divisão
    )
    
fig2.update_traces(marker=dict(line=dict(color='rgb(8,48,107)', width=0.8))) # fatias destacadas a preto

fig2.update_layout(
    title = dict(
        text = '<b>Distribuição percentual de medalhistas por sexo e estação em diferentes anos</b>',
        font = dict(color = '#1E88E5', size = 27, family = "Calibri"),
        x = 0.43
    ),
    legend_title = dict(text = 'Categorias:', font = dict(size = 14))
)

fig2.show()



fig3 = go.Figure(data = [
    go.Bar(x = winter_counts.index, y = summer_counts['Homens'], name = 'Verão (Homens)', marker_color = '#2196f3', offsetgroup = "Summer"),
    go.Bar(x = winter_counts.index, y = summer_counts['Mulheres'], name = 'Verão (Mulheres)', marker_color = '#f74780', offsetgroup = "Summer"),
    go.Bar(x = winter_counts.index, y = winter_counts['Homens'], name = 'Inverno (Homens)', marker_color = '#c3e3fd', offsetgroup = "winter"),
    go.Bar(x = winter_counts.index, y = winter_counts['Mulheres'], name = 'Inverno (Mulheres)', marker_color = '#ffc1d5', offsetgroup = "winter")

])

fig3.update_traces(marker_line_color = 'rgb(8,48,107)', marker_line_width = 0.3) # linhas pretas à volta das barras

fig3.update_layout(
    barmode = 'stack',  # barras empilhados em cada ano para cada grupo (summer e winter)
    title = dict(
        text = '<b>Comparação anual de medalhistas por estação e sexo</b>',
        font = dict(color = '#1E88E5', size = 27, family = "Calibri")
        #x = 0.4
    ),   
    xaxis = dict(
        title = dict(text = 'Ano', font = dict(size = 16, color = "#020202")),
        gridcolor = '#FFB5C0',
        tickvals = winter_counts.index, 
        tickangle = -40  # rodar ligeiramente os anos para se perceberem melhor
    ),
    yaxis = dict(
        title = dict(text = 'Número de Medalhistas', font = dict(size = 16, color = "#020202")),
        gridcolor = '#FFB5C0'
    ),
    legend_title = dict(text = 'Categorias:', font = dict(size = 14)),
    plot_bgcolor = "#FFF9FA"  # cor de fundo do gráfico #fdf1f1
)

fig3.show()




# fazer uma cópia para não afetar as células anteriores
winter_copy = winter_counts.copy()
summer_copy = summer_counts.copy()

# Acrescentar os anos sem provas para visualizar esse intervalo sem jogos
summer_copy.loc[1940] = [0, 0]
summer_copy.loc[1944] = [0, 0]
winter_copy.loc[1940] = [0, 0]
winter_copy.loc[1944] = [0, 0]

# Reordenar os índices por ano
summer_copy = summer_copy.sort_index()
winter_copy = winter_copy.sort_index()


fig4 = go.Figure()

# ---- Verão ----
fig4.add_trace(go.Barpolar(
    r = summer_copy['Homens'], # eixo do raio (nº de medalhistas)
    theta = winter_copy.index.astype(str), # eixo circular (anos)
    name = 'Verão (Homens)',
    marker = dict(     
        color = '#2196f3',
        line = dict(color = 'rgb(8,48,107)', width = 0.3) # linha à volta das fatias
    )
))

fig4.add_trace(go.Barpolar(
    r = summer_copy['Mulheres'],
    theta = winter_copy.index.astype(str),
    name = 'Verão (Mulheres)',
    marker = dict(
        color = '#f74780',
        line = dict(color = 'rgb(8,48,107)', width = 0.3)
    )
))

# ---- Inverno ----
fig4.add_trace(go.Barpolar(
    r = winter_copy['Homens'],
    theta = winter_copy.index.astype(str),
    name = 'Inverno (Homens)',
    marker = dict(
        color = '#c3e3fd',
        line = dict(color = 'rgb(8,48,107)', width = 0.3)
    )
))

fig4.add_trace(go.Barpolar(
    r = winter_copy['Mulheres'],
    theta = winter_copy.index.astype(str),
    name = 'Inverno (Mulheres)',
    marker = dict(
        color = '#ffc1d5',
        line = dict(color = 'rgb(8,48,107)', width = 0.3)
    )
))

fig4.update_layout(
    title = dict(text = '<b>Crescimento global do número de medalhistas ao longo dos anos</b>', 
                 font = dict(color = '#1E88E5', size = 27, family = "Calibri"),
                 x = 0.4
    ),
    legend_title = dict(text = 'Categorias:', font = dict(size = 14)),
    polar = dict(
        # eixo no raio do círculo
        radialaxis = dict(
            title = '<b>Nº DE MEDALHISTAS<b>', 
            color = "#000080", 
            linecolor = "#000080",  
            tickfont = dict(color = "#000080", size = 10, family = "Arial Black"), gridcolor = '#FFB5C0'), 
        # datas na eixo circular
        angularaxis = dict(    
            direction = 'clockwise',
            # lista das posições angulares
            tickvals = [1924, 1928, 1932, 1936, 0, 0, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 1984, 1988, 1992], # 1940, 1944 não têm provas
            ticktext = [                                
                    f"<span style='font-weight:bold; font-size:{size}px; color:#000080'>{year}</span>" # valores a negrito e com tamaho decrescente
                    for size, year in zip([20, 18, 17.7, 17.2, 16.8, 16.2, 15.6, 15, 14.5, 14, 13.5, 13, 12.5, 12.5, 12, 11.7, 11.5, 11.5], winter_copy.index)
                ],
            gridcolor = '#FFB5C0'
        ), 
        bgcolor = "#FFF9FA" # cor do fundo do gráfico polar #fdf1f1
))

fig4.show()




fig5 = go.Figure(data = [
    go.Bar(x = winter_counts.index, y = summer_counts['Homens'], name = 'Verão (Homens)', marker_color = '#2196f3'),
    go.Bar(x = winter_counts.index, y = summer_counts['Mulheres'], name = 'Verão (Mulheres)', marker_color = '#f74780'),
    go.Bar(x = winter_counts.index, y = winter_counts['Homens'], name = 'Inverno (Homens)', marker_color = '#c3e3fd'),
    go.Bar(x = winter_counts.index, y = winter_counts['Mulheres'], name = 'Inverno (Mulheres)', marker_color = '#ffc1d5')

])

fig5.update_traces(marker_line_color = 'rgb(8,48,107)', marker_line_width = 0.3) # linhas pretas à volta das barras

fig5.update_layout(
    barmode = 'stack',  # barras (dos homens e mulheres de cada estação) empilhados em cada ano
    title = dict(
        text = '<b>Distribuição Total de Medalhistas por Sexo e Estações do Ano desde 1924 até 1992</b>',
        font = dict(color = '#1E88E5', size = 27, family = "Calibri")
        #x = 0.4
    ),   
    xaxis = dict(
        title = dict(text = 'Ano', font = dict(size = 16, color = "#020202")),
        gridcolor = '#FFB5C0',
        tickvals = winter_counts.index, 
        tickangle = -40  # rodar ligeiramente os anos para se perceberem melhor
    ),
    yaxis = dict(
        title = dict(text = 'Número de Medalhistas', font = dict(size = 16, color = "#020202")),
        gridcolor = '#FFB5C0'
    ),
    legend_title = dict(text = 'Categorias:', font = dict(size = 14)),
    plot_bgcolor = "#FFF9FA"  # cor de fundo do gráfico #fdf1f1
)

fig5.show()