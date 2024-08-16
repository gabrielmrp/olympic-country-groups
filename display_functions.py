import os
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
import streamlit as st

CHART_WIDTH = 16
CHART_HEIGHT = 9
DARK_INNER_COLOR = '#222'
MAP_BORDER_COLOR = '#888'

FONT_TITLE = 20
FONT_X_LABEL = 14
FONT_Y_LABEL = 14
FONT_LEGEND = 11
FONT_LEGEND_BAR = 14
FONT_NUMBERS = 24

plt.style.use('dark_background')



def elbow_chart(df_by_country,show):  

    if not show:
        return
    # Lista para armazenar a soma das distâncias quadradas dentro dos clusters (WSS)
    wss = [] 
    plt.rcParams['axes.grid'] = True 
    plt.rcParams["grid.linestyle"] = "--"

    # Testan valores de k de 1 a 10
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(df_by_country)
        # inertia_ é a soma das distâncias quadradas dentro dos clusters
        wss.append(kmeans.inertia_)  

    
    # Plota o gráfico do cotovelo
    plt.figure(figsize=(CHART_WIDTH,CHART_HEIGHT))
    plt.plot(range(1, 11), wss, marker='o', linewidth=3)
    plt.plot(range(8, 9), wss[7], marker='.', markersize=25 , color='yellow')
    plt.gca().set_facecolor(DARK_INNER_COLOR)
    plt.title('Gráfico do Cotovelo',fontsize = FONT_TITLE * 1.5)
    plt.xlabel('Número de Clusters',fontsize = FONT_X_LABEL * 1.5)
    plt.ylabel('Soma das dist. quadradas nos clusters',fontsize = FONT_Y_LABEL * 1.5)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()


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

    # Plotar cada PCA com cores diferentes
    colors = ['b', 'g', 'r']

    pca_titles = pca_titles[:3]

    bar_width = 0.25

    # Posicionamento as barras
    indices = np.arange(len(loadings_df.index))

    # Plota a contribuição das features para PC1 e PC2
    plt.figure(figsize=(CHART_WIDTH,CHART_HEIGHT))
    plt.gca().set_facecolor(DARK_INNER_COLOR)

    # Plotar as barras para cada componente
    for i, pcax in enumerate(pca_titles):
        if i < 3:  # Limita a 3 componentes principais para o exemplo
            plt.bar(indices + i * bar_width, loadings_df[pcax], bar_width,
                     color=colors[i], label=f'Contribuição para {pcax}', alpha=0.7)

    # Ajuste do gráfico
    plt.title('Contribuição das Features para PC1, PC2 e PC3', pad=20, fontsize=FONT_TITLE * 1.5)
    plt.xlabel('Features',fontsize = FONT_X_LABEL * 1.5)
    plt.ylabel('Coeficiente',fontsize = FONT_Y_LABEL * 1.5)
    plt.xticks(indices + bar_width, loadings_df.index, rotation=45, ha='right',fontsize = FONT_X_LABEL * 1.5)
    plt.legend(fontsize = FONT_LEGEND_BAR , loc = 'lower left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(plt)
    plt.close()

    st.dataframe(loadings_df.T.head(1).round(1), use_container_width=True)



def print_23d_clusters(df_by_country,scaled_features,name_symbol_dict,labels,dimension,show=False):

    if not show:
        return
    
    # Reduz para 3 dimensões usando PCA para visualização
    pca = PCA(n_components=3) 
    principal_components = pca.fit_transform(scaled_features)

    # Adiciona as componentes principais ao DataFrame
    df_by_country['PC1'] = principal_components[:, 0]
    df_by_country['PC2'] = principal_components[:, 1]
    if dimension >2:
        df_by_country['PC3'] = principal_components[:, 2]

    # Exibe a variância explicada por cada componente
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance) 
    
    # As cores estão nessa ordem para coincidirem com o mapa, por isso esse vetor é redundante
    # TODO entender a lógica de ordem do dois gráficos que utilizam as cores e compatibilizar

    colors_2d = ['orange','purple' , 'red', 'whitesmoke', 'blue', 'brown', 'yellow','green' ]
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'yellow', 'whitesmoke']

    
    # Exibe gráfico interativo com Plotly 
     

    if dimension > 2:
        df_by_country['Cluster_Labels'] = df_by_country['Cluster'].map(lambda x: labels[x])
    else:
        df_by_country = df_by_country.sort_values(by='Cluster')
        df_by_country['Cluster_Labels'] = df_by_country['Cluster'].map(lambda x: f'Cluster {x+1}')

    df_by_country['Cluster_Colors'] = df_by_country['Cluster'].map(lambda x: colors[x])

    if dimension > 2:
        fig = px.scatter_3d(df_by_country, x='PC1', y='PC2', z='PC3', color='Cluster_Labels', text='Symbol',
                        hover_name=df_by_country.Symbol.map({v: k for k, v in name_symbol_dict.items()}),
                        title='Clusters representados até a 3ª aplicação de PCA: PC3',
                        labels={'PC1': 'PC1', 'PC2': 'PC2', 'PC3': 'PC3', 'Cluster_Labels': 'Grupos'},
                        color_discrete_sequence={i:colors[i] for i in range(len(colors))})
        fig.update_layout(
            legend_font_size=10,
            font=dict(
            family="Arial, sans-serif",  # Fonte do texto
            size=12 
        ))
    else:
        fig = px.scatter(df_by_country, x='PC1', y='PC2', color='Cluster_Labels', text='Symbol',
                        hover_name=df_by_country.Symbol.map({v: k for k, v in name_symbol_dict.items()}),
                        title='Clusters representados até a 2ª aplicação de PCA: PC2',
                        labels={'PC1': 'PC1', 'PC2': 'PC2', 'Cluster_Labels': 'Grupos'},
                        color_discrete_sequence={i:colors_2d[i] for i in range(len(colors_2d))})
        fig.update_layout(
            legend_font_size=10,
            font=dict(
            family="Arial, sans-serif",  # Fonte do texto
            size=8 
        ))

    # Atualiza os textos para exibir os símbolos
    fig.update_traces(textposition='top center')
    #fig.show()
    st.plotly_chart(fig)

    if dimension > 2:
        df_cumulative_variance = pd.DataFrame({'Variância explicada cumulativa': cumulative_variance
                                    }, index=[f'PC{i+1}' for i in range(len(cumulative_variance))])

        # Arredondar os valores
        df_cumulative_variance = df_cumulative_variance.round(2).map(lambda x: str(round(x * 100)) + '%').T

        st.dataframe(df_cumulative_variance , use_container_width=True)
    

def create_cluster_feature_heatmap(df_features,feature_names,df_by_country,labels,show=False):

    if not show:
        return
    # Calcula as médias para cada cluster e para os outros clusters
    cluster_means = [] 

    for cluster_id in range(len(df_by_country.Cluster.unique())):

        # Índices onde cluster == cluster_id
        indices_cluster = df_by_country[df_by_country['Cluster'] == cluster_id].index

        # Média das features para cluster == cluster_id
        mean_cluster = df_by_country.loc[indices_cluster, df_features.columns].mean() 

        # Média das features para cluster != cluster_id 
        mean_all_clusters = df_by_country.loc[:, df_features.columns].mean()

        # Adiciona as médias à lista
        cluster_means.append([mean_cluster / mean_all_clusters])

    # Cria um DataFrame para o heatmap
    heatmap_data = pd.DataFrame()
    for i, means in enumerate(cluster_means):
        heatmap_data[f'C{i}: {labels[i]}'] = means[0]

    heatmap_data.index = feature_names
    heatmap_data = heatmap_data.T

    # Plota o heatmap
    plt.figure(figsize=(CHART_WIDTH,CHART_HEIGHT))
    sns.set_context("notebook", font_scale=1.5)
    ax = sns.heatmap(heatmap_data, annot=True, cmap='coolwarm',
                 fmt='.1f', annot_kws={"fontsize": FONT_NUMBERS})
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=FONT_X_LABEL * 1.5, color='white')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=FONT_Y_LABEL * 1.5, color='white')
    
    plt.title('Médias das Features por Cluster',fontsize = FONT_TITLE * 1.5)

    st.pyplot(plt)
    plt.close()


def generate_map(world,quant_clusters,labels,show=False):

    plt.style.use('Solarize_Light2')
     
    plt.rcParams['lines.linewidth']=1
    plt.rcParams['axes.facecolor']='#282828'
    plt.rcParams['savefig.facecolor']='#282828'
    plt.rcParams['axes.grid'] = False  
    plt.rcParams['xtick.labelsize'] = 0 
    plt.rcParams['ytick.labelsize'] = 0
    plt.rcParams['axes.linewidth'] = 0 
    plt.rcParams['xtick.major.size'] = 0
    plt.rcParams['ytick.major.size'] = 0


    # As cores estão nessa ordem para coincidirem com o plot de clusters, por isso esse vetor é redundante
    # TODO entender a lógica de ordem do dois gráficos que utilizam as cores e compatibilizar
    colors = ['orange','yellow','red','whitesmoke','blue','brown','purple','green']

    # Cria a visualização
    fig, _ = plt.subplots(figsize=(16,9))

    # Cria um subplot para o mapa e um para a legenda
    ax_map = fig.add_subplot(111)

    #fig.style.use('dark_background')

    # Remove eixos do mapa e definir fundo branco
    ax_map.set_axis_off() 
    ax_map.grid(True, linestyle='--', alpha=0)
    ax_map.set_facecolor('red')

    # Ajustar a posição do subplot da legenda para ficar mais próximo do mapa
     # [left, bottom, width, height]
    ax_legend = fig.add_axes([0.11, 0.135, 0.8, 0.03])
    ax_legend.set_axis_off()
    ax_legend.set_facecolor('red')

    # Plota o mapa
    world.boundary.plot(ax=ax_map, color=MAP_BORDER_COLOR)

    # Plota cada grupo de acordo com a cor correspondente
    for i in range(quant_clusters):
        group_data = world[world['Group'] == i]
        if not group_data.empty:
            group_data.plot(ax=ax_map, color=colors[i])

    ax_map.set_title('Mapa dos grupos de países de acordo com características olímpicas',
                     color='white',
                     fontsize=FONT_TITLE)  

    # Adicionar a legenda em um subplot separado
    patches = [Patch(color=colors[i], label=labels[i]) for i in range(quant_clusters)]
    ax_legend.legend(handles=patches, loc='center', bbox_to_anchor=(0.5, 0.5), ncol=4, fontsize = FONT_LEGEND * 1.125)


    if show:
        st.pyplot(plt)
    fig.savefig('mapa_dos_grupos_de_paises.png', dpi=300, bbox_inches='tight', facecolor='#d1d1d1')
    #plt.rcParams = plt_rc

    plt.rcParams['axes.grid'] = True   
    plt.rcParams['xtick.labelsize'] = 16.5
    plt.rcParams['ytick.labelsize'] = 16.5
    plt.rcParams['axes.linewidth'] = 1.25
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['xtick.major.size'] = 6


def crop_image(show=False):
    filename = 'mapa_dos_grupos_de_paises'
    
    try:
        # Abrir a imagem original
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
        cropped_image.save(f'{filename}_cropped.png')
        
    except Exception as e:
        # Se houver um erro ao abrir ou cortar a imagem, exibir a imagem original
        st.error(f"Erro ao processar a imagem: {e}")
        cropped_image = image
    
    # Exibir a imagem cortada em Streamlit
    if show:
        st.image(cropped_image, 
                 caption='Mapa dos grupos de países de acordo com características olímpicas',
                 use_column_width=True)