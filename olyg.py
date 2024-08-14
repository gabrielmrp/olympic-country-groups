import warnings
import numpy as np
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import streamlit as st
from features_functions import def_gender,get_feature_total_medals,get_feature_gold_medals,get_feature_gender,get_feature_collective,get_feature_categories_diverse
from display_functions import print_3d_clusters,show_feature_to_pca,create_cluster_feature_heatmap,generate_map,crop_image,elbow_chart
    

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*geopandas.dataset.*")


categories = {
"Atletismo":"Atletismo","Badminton":"Bola","Basquete":"Bola","Basquete 3x3":"Bola",
"Boxe":"Luta","Breaking":"Radicais","Canoagem de Velocidade":"Aquáticos",
"Canoagem Slalom":"Aquáticos","Ciclismo BMX Freestyle":"Ciclismo",
"Ciclismo BMX Racing":"Ciclismo","Ciclismo de Estrada":"Ciclismo",
"Ciclismo de Pista":"Ciclismo","Ciclismo Mountain Bike":"Ciclismo",
"Escalada":"Precisão","Escalada Esportiva":"Precisão","Esgrima":"Luta","Futebol":"Bola","Ginástica Artística":"Artístico",
"Ginástica de Trampolim":"Artístico","Ginástica Rítmica":"Artístico","Golfe":"Precisão","Handebol":"Bola",
"Hipismo":"Hipismo","Hóquei sobre Grama":"Bola","Judô":"Luta","Levantamento de Peso":"Atletismo",
"Luta":"Luta","Maratona Aquática":"Natação","Nado Artístico":"Artístico","Natação":"Natação",
"Pentatlo Moderno":"Completo","Polo Aquático":"Bola","Remo":"Aquáticos","Rugby Sevens":"Bola",
"Saltos Ornamentais":"Artístico","Skate":"Radicais","Surfe":"Radicais","Taekwondo":"Luta",
"Tênis":"Bola","Tênis de Mesa":"Bola","Tiro com Arco":"Precisão","Tiro Esportivo":"Precisão",
"Triatlo":"Completo","Vela":"Aquáticos","Vôlei":"Bola","Vôlei de Praia":"Bola",
"Canoagem Velocidade":"Aquáticos"
}

pd.set_option('display.max_rows', None)


def apply_kmeans(optimal_k,df_features,name_symbol_dict):
    # Selecionar todas as colunas numéricas 
    # Normalizar os dados
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)

    # Aplicar o K-means
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    clusters = kmeans.fit_predict(scaled_features)

    return clusters,scaled_features



def map_symbols_to_clusters(df_by_country,name_symbol_dict):
    df_sym_cluster = pd.DataFrame()
    for i in range(len(df_by_country.Cluster.value_counts())):
        aux = pd.DataFrame(df_by_country[df_by_country.Cluster == i]['name'].map(name_symbol_dict)).set_index('name')
        aux['Cluster']  = i
        df_sym_cluster = pd.concat([df_sym_cluster,aux])
    df_sym_cluster=df_sym_cluster.reset_index()

    return df_sym_cluster



def fill_groups(df_sym_cluster):

    # Baixar o shapefile do mundo
    # TODO utilizar outra forma de obter o modelo, pois só funciona na versão <1 do geopandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


    # Dicionário de substituição de códigos ISO
    iso_a3_rep = {'FIJ':'FJI', 'NED':'NLD', 'GER':'DEU', 'KOS':'-99', 'CRO': 'HRV', 'MGL': 'MNG', 'TPE': 'TWN', 'PHI': 'PHL',
                  'BOT': 'BWA', 'RSA': 'ZAF', 'BUL': 'BGR', 'POR': 'PRT', 'SUI': 'CHE', 'GRE': 'GRC', 'IRI': 'IRL', 'MAS': 'MYS',
                  'PUR': 'PRI', 'DEN': 'DNK', 'ZAM': 'ZMB', 'SLO': 'SVN', 'ALG': 'DZA', 'INA': 'IDN', 'CHI': 'CHL', 'GUA': 'GTM'}

    excluded = ['HKG', 'SGP', 'EOR', 'LCA', 'GRN', 'CPV', 'DMA']

    # Adicionar colunas de grupo
    world['Group'] = None

    # Mapear os países para o mundo
    for _,row in df_sym_cluster.iterrows():
        country = row['name']
        group = row['Cluster']
        if country not in excluded and (country in world['iso_a3'].values or country in iso_a3_rep.keys()):
            if country in iso_a3_rep.keys():
                country = iso_a3_rep[country]
            world.loc[world['iso_a3'] == country, 'Group'] = group
        else:
            if country in excluded:
                pass
            else:
                pass
                #print(country)

    return world

if __name__ == "__main__":

    st.write('hello')
    # Abre o dataframe com todos os dados
    df = pd.read_excel('olympic_medal_table_2024_v1.xlsx')

    # Lê os nomes dos grupos
    with open('labels.json', 'r', encoding='utf-8') as file:
        labels = json.load(file)
 
    # Lê os nomes das features
    with open('features.json', 'r', encoding='utf-8') as file:
        feature_names = json.load(file)    

    # Extrai os detalhes das features
    df['categories'] = df['sport'].map(categories)
    df['gender'] = df.modality.str.lower().map(def_gender)
    df['collective'] = df['athlete'].map({'Team':'team'}).fillna('individual')

    # Cria o dataframe por pais 
    df_by_country = df[['name','gold','total_medals']].drop_duplicates().reset_index(drop=True)

    # Coleta as features
    get_feature_total_medals(df_by_country)
    get_feature_gold_medals(df_by_country)

    df_by_country.drop(['total_medals','gold'], axis=1, inplace=True)

    df_by_country = get_feature_gender(df,df_by_country)
    df_by_country = get_feature_collective(df,df_by_country)
    df_by_country = get_feature_categories_diverse(df,df_by_country)

    # Cria os dicionários de conversão de símbolo para nome de país e vice-versa
    name_symbol_dict = df.groupby(['name', 'symbol']).count()['sport'].reset_index()[['name', 'symbol']].set_index('name').to_dict()['symbol']
    symbol_name_dict = {v: k for k, v in name_symbol_dict.items()}

    show = False

    df_features = df_by_country.copy()
    df_features = df_features.reset_index().drop('index', axis=1).set_index('name') 
    df_features.index = df_features.index.map(name_symbol_dict)

    # Utiliza o gráfico do cotovelo para decidir o número de clusters
    elbow_chart(df_features,show)

    # Número de clusters escolhido (exemplo: 8)
    optimal_k = 8
    
    # Aplica o método de clusterização
    clusters,scaled_features  = apply_kmeans(optimal_k,df_features,name_symbol_dict)

    # Adicionar os clusters ao DataFrame de países
    df_by_country['Cluster'] = clusters
    df_by_country['Symbol'] = df_features.index
    
    # Mostra o critério de relevância de feature por cluster
    show_feature_to_pca(scaled_features,feature_names,show)
    

    # Mostra os cluster em um espaço 3d
    print_3d_clusters(df_by_country,scaled_features,name_symbol_dict,labels,show)
 
    
    # Mostra o heatmat de relevância de feature por cluster
    create_cluster_feature_heatmap(df_features,feature_names,df_by_country,show)
    

    # Compatibilização do símbolo para gerar o mapa
    world = fill_groups(map_symbols_to_clusters(df_by_country,name_symbol_dict))
    
    show = True
    generate_map(world,optimal_k,labels,show)
    show = False

    # TODO remover os eixos sem precisar cortar as imagens
    crop_image(show)
