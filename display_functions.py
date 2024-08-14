import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt 
from PIL import Image
import seaborn as sns
from matplotlib.patches import Patch
CHART_WIDTH = 16
CHART_HEIGHT = 9
DARK_INNER_COLOR = '#222'
MAP_BORDER_COLOR = '#d1d1d1'
plt.style.use('dark_background')

def print_3d_clusters(df_by_country,scaled_features,name_symbol_dict,labels,show=False):

    if not show:
        return
    
    # Reduz para 3 dimensões usando PCA para visualização
    pca = PCA(n_components=3) 
    principal_components = pca.fit_transform(scaled_features)

    # Adiciona as componentes principais ao DataFrame
    df_by_country['PCA1'] = principal_components[:, 0]
    df_by_country['PCA2'] = principal_components[:, 1]
    df_by_country['PCA3'] = principal_components[:, 2]

    # Exibe a variância explicada por cada componente
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance) 

    # As cores estão nessa ordem para coincidirem com o mapa, por isso esse vetor é redundante
    # TODO entender a lógica de ordem do dois gráficos que utilizam as cores e compatibilizar

    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'pink', 'brown']
    
    # Exibe gráfico interativo com Plotly 
    
    color_discrete_map={i:colors[i] for i in range(len(colors))}

    df_by_country['Cluster_Labels'] = df_by_country['Cluster'].map(lambda x: labels[x])
    df_by_country['Cluster_Colors'] = df_by_country['Cluster'].map(lambda x: colors[x])

    fig = px.scatter_3d(df_by_country, x='PCA1', y='PCA2', z='PCA3', color='Cluster_Labels', text='Symbol',
                        hover_name=df_by_country.Symbol.map({v: k for k, v in name_symbol_dict.items()}),
                        title='Clusters com PCA',
                        labels={'PCA1': 'PCA1', 'PCA2': 'PCA2', 'PCA3': 'PCA3', 'Cluster_Labels': 'Grupos'},
                        color_discrete_sequence=color_discrete_map)

    # Atualiza os textos para exibir os símbolos
    fig.update_traces(textposition='top center')
    fig.show()



def show_feature_to_pca(scaled_features,feature_names,show=False):

    if not show:
        return 

    # Aplica o PCA
    pca = PCA()
    pca.fit(scaled_features)

    # Coeficientes das componentes principais
    loadings = pca.components_

    pca_titles = [f'PC{i+1}' for i in range(loadings.shape[0])]

    # Criar o DataFrame com os coeficientes
    loadings_df = pd.DataFrame(loadings.T, columns=pca_titles, index=feature_names)

    # Mostra os coeficientes
    print("Coeficientes das Componentes Principais:")

    # Plota a contribuição das features para PCA1 e PCA2
    plt.figure(figsize=(CHART_WIDTH,CHART_HEIGHT))

    plt.gca().set_facecolor(DARK_INNER_COLOR)  # Fundo preto

    # Plotar cada PCA com cores diferentes
    colors = ['b', 'g', 'r']

    pca_titles = pca_titles[:3]

    bar_width = 0.25

    # Posicionamento as barras
    indices = np.arange(len(loadings_df.index))

    # Plotar as barras para cada componente
    for i, pcax in enumerate(pca_titles):
        if i < 3:  # Limita a 3 componentes principais para o exemplo
            plt.bar(indices + i * bar_width, loadings_df[pcax], bar_width, color=colors[i], label=f'Contribuição para {pcax}', alpha=0.7)

    # Ajuste do gráfico
    plt.title('Contribuição das Features para PCA1, PCA2 e PCA3', pad=20, fontsize=16)
    plt.xlabel('Features')
    plt.ylabel('Coeficiente')
    plt.xticks(indices + bar_width, loadings_df.index, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def create_cluster_feature_heatmap(df_features,feature_names,df_by_country,show=False):

    if not show:
        return
    # Calcula as médias para cada cluster e para os outros clusters
    cluster_means = []

    for cluster_id in range(len(df_by_country.Cluster.unique())):

        # Índices onde cluster == cluster_id
        indices_cluster = df_by_country[df_by_country['Cluster'] == cluster_id].index

        # Média das features para cluster == cluster_id
        mean_cluster = df_by_country.loc[indices_cluster, df_features.columns].mean()

        # Índices onde cluster != cluster_id
        indices_other_clusters = df_by_country[df_by_country['Cluster'] != cluster_id].index

        # Média das features para cluster != cluster_id
        mean_other_clusters = df_by_country.loc[indices_other_clusters, df_features.columns].mean()

        # Adiciona as médias à lista
        cluster_means.append([mean_cluster / mean_other_clusters])

    # Cria um DataFrame para o heatmap
    heatmap_data = pd.DataFrame()
    for i, means in enumerate(cluster_means):
        heatmap_data[f'Cluster {i}'] = means[0]

    heatmap_data.index = feature_names

    # Plota o heatmap
    plt.figure(figsize=(CHART_WIDTH,CHART_HEIGHT))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Médias das Features por Cluster')
    plt.show()

def elbow_chart(df_by_country,show):  

    if not show:
        return
    # Lista para armazenar a soma das distâncias quadradas dentro dos clusters (WSS)
    wss = [] 

    # Testan valores de k de 1 a 10
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(df_by_country)
        wss.append(kmeans.inertia_)  # inertia_ é a soma das distâncias quadradas dentro dos clusters

    
    # Plota o gráfico do cotovelo
    plt.figure(figsize=(CHART_WIDTH,CHART_HEIGHT))
    plt.plot(range(1, 11), wss, marker='o')
    plt.title('Gráfico do Cotovelo')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Soma das Distâncias Quadradas dentro dos Clusters')
    plt.show()


def generate_map(world,quant_clusters,labels,show=False):

    # As cores estão nessa ordem para coincidirem com o plot de clusters, por isso esse vetor é redundante
    # TODO entender a lógica de ordem do dois gráficos que utilizam as cores e compatibilizar
    colors = ['orange','pink','red','grey','blue','cyan','purple','green']

    # Cria a visualização
    fig, _ = plt.subplots(figsize=(CHART_WIDTH,CHART_HEIGHT))

    # Cria um subplot para o mapa e um para a legenda
    ax_map = fig.add_subplot(111)

    # Remove eixos do mapa e definir fundo branco
    ax_map.set_axis_off()
    ax_map.set_facecolor(DARK_INNER_COLOR)

    # Ajustar a posição do subplot da legenda para ficar mais próximo do mapa
     # [left, bottom, width, height]
    ax_legend = fig.add_axes([0.1, 0.135, 0.8, 0.03])
    ax_legend.set_axis_off()
    ax_legend.set_facecolor(DARK_INNER_COLOR)

    # Plota o mapa
    world.boundary.plot(ax=ax_map, color=MAP_BORDER_COLOR)

    # Plota cada grupo de acordo com a cor correspondente
    for i in range(quant_clusters):
        group_data = world[world['Group'] == i]
        if not group_data.empty:
            group_data.plot(ax=ax_map, color=colors[i])

    ax_map.set_title('Mapa dos grupos de países de acordo com características olímpicas', color='black')  

    # Adicionar a legenda em um subplot separado
    patches = [Patch(color=colors[i], label=labels[i]) for i in range(quant_clusters)]
    ax_legend.legend(handles=patches, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=4)


    if show:
        plt.show()
    fig.savefig('mapa_dos_grupos_de_paises.png', dpi=300, bbox_inches='tight')


def crop_image(show=False):

    # TODO remover essa função assim que a remoção dos eixos no mapa funcionar
    
    filename='mapa_dos_grupos_de_paises'

    image = Image.open(f'{filename}.png')

    # Obter dimensões da imagem
    width, height = image.size

    # Calcular os valores de corte
    left = width * 0.07
    top = height * 0.03
    right = width * 0.95
    bottom = height * 0.95

    # Cortar a imagem
    cropped_image = image.crop((left, top, right, bottom))

    # Salvar a imagem cortada
    cropped_image.save(f'{filename}_cropped.png')

    if show:
        cropped_image.show()
