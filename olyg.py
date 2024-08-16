import warnings
import numpy as np
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import streamlit as st
from features_functions import def_gender,get_feature_total_medals,get_feature_gold_medals,get_feature_gender,get_feature_collective,get_feature_categories_diverse
from display_functions import print_23d_clusters,show_feature_to_pca,create_cluster_feature_heatmap,generate_map,crop_image,elbow_chart
from st_aggrid import AgGrid, GridOptionsBuilder
    

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*geopandas.dataset.*")

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



def map_symbols_to_clusters(df_by_country,labels,name_symbol_dict):
    df_sym_cluster = pd.DataFrame()
    dict_country_label = { l:[] for l in labels }

    for i in range(len(df_by_country.Cluster.value_counts())):

        dict_country_label[labels[i]] = df_by_country[df_by_country.Cluster == i].sort_values(['name'])['name'].tolist()
        #for country in range(pd.DataFrame(df_by_country[df_by_country.Cluster == i])):
        #    #st.write(country)
        #    #dict_country_label[labels[i]].append(country['name'])

        aux = pd.DataFrame(df_by_country[df_by_country.Cluster == i]['name'].map(name_symbol_dict)).set_index('name')
        aux['Cluster']  = i
        df_sym_cluster = pd.concat([df_sym_cluster,aux])
    df_sym_cluster=df_sym_cluster.reset_index()
 
    return df_sym_cluster,dict_country_label



def fill_groups(df_sym_cluster):

    # Baixar o shapefile do mundo
    # TODO utilizar outra forma de obter o modelo, pois só funciona na versão <1 do geopandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


    # Dicionário de substituição de códigos ISO
    with open('config/symbols.json', 'r', encoding='utf-8') as file:
        symbols_rep = json.load(file)

    excluded = ['HKG', 'SGP', 'EOR', 'LCA', 'GRN', 'CPV', 'DMA']

    # Adicionar colunas de grupo
    world['Group'] = None

    # Mapear os países para o mundo
    for _,row in df_sym_cluster.iterrows():
        country = row['name']
        group = row['Cluster']
        if country not in excluded and (country in world['iso_a3'].values or country in symbols_rep.keys()):
            if country in symbols_rep.keys():
                country = symbols_rep[country]
            world.loc[world['iso_a3'] == country, 'Group'] = group

        else:
            if country in excluded:
                pass
            else:
                pass
                #print(country)

    return world

if __name__ == "__main__": 

    st.set_page_config(page_title="Perfis de Países de Acordo com o Desempenho Olímpico", page_icon="🥇")
    show = True

    st.markdown(
    """
    <style>
        .label_0{color:orange}
        .label_1{color:yellow}
        .label_2{color:red}
        .label_3{color:whitesmoke}
        .label_4{color:blue}
        .label_5{color:brown}
        .label_6{color:purple}
        .label_7{color:green}
        .country{color:#a1a1a1!important}
        ol {
            list-style-type: decimal; /* Ensure list items are numbered */
        }
        *{text-align: justify;}
        h1{text-align:center}
        h2,h3{text-align:left}
        .techinfo{color: lightblue;margin-left:2em}
        .alert{color: yellow}
        ul { margin: 0}

    </style>

    """,
    unsafe_allow_html=True) 
 
 
    # Lê os nomes das categorias
    with open('config/categories.json', 'r', encoding='utf-8') as file:
        categories = json.load(file)


    # Lê os nomes dos grupos
    with open('config/labels.json', 'r', encoding='utf-8') as file:
        labels = json.load(file)
 
    # Lê os nomes das features
    with open('config/features.json', 'r', encoding='utf-8') as file:
        feature_names = json.load(file)  

    st.title('Perfis de Países de Acordo com o Desempenho Olímpico')

    st.write('Neste projeto de ciência de dados, utilizamos um algoritmo de clusterização para identificar grupos de delegações esportivas nas Olimpíadas de Paris 2024, com base em características de desempenho.')

    st.write('Inicialmente, detalharemos o banco de dados utilizado e seu processo de construção. Em seguida, mostraremos as características usadas para identificar os perfis. Posteriormente, aplicaremos o algoritmo de clusterização e apresentaremos os grupos formados por meio de gráficos de dispersão e de um Mapa.')

    st.markdown('<p class="alert">⚠️ Importante, Importante: o ícone ✍️ indica que a leitura destacada contém detalhes mais técnicos, podendo ser pulada sem prejuízo para o entendimento geral.', unsafe_allow_html=True)

    st.markdown('<p class="techinfo">✍️ Para esse projeto utilizamos a linguagem Python e as bibliotecas: pandas e geopandas (manipulação de base de dados) scikit-learn (machine learning); plotly, matplotlib e seaborn (visualização) e também o streamlit para a apresentação. O código se encontra ao final da página.</p>', unsafe_allow_html=True)

    st.divider()
    st.header('Base de dados:')
    st.markdown('A base de dados foi montada a partir da coleta dos dados diretamente do <a href="https://olympics.com/pt/paris-2024/medalhas">site oficial das Olimpíadas Paris 2024</a>.', unsafe_allow_html=True)

    st.markdown('<p class="techinfo">✍️ O processo de web scraping para a coleta dos dados foi realizado de maneira semi-automática, e seu detalhamento está além do escopo desta apresentação.</p>', unsafe_allow_html=True)

 

    # Abre o dataframe com todos os dados
    df = pd.read_excel('olympic_medal_table_2024_v1.xlsx', index_col=0)

    st.write('Os dados apresentados na tabela de medalhas são organizados em três níveis: Países, Esporte e Medalha, com cada linha representando uma medalha. Abaixo, vamos exemplificar esses níveis usando como exemplo a medalha de ouro de Rebeca Andrade no solo.')

    # Lê os nomes dos países a serem substituidos

    def st_show_dataframe_compacted(series_aux, columns):
        df_aux = pd.DataFrame(series_aux).T
        df_aux = df_aux[columns]
        st.dataframe(df_aux)

 
    st.write('País')   

    st_show_dataframe_compacted(df.iloc[722],['symbol','name','gold','silver','bronze','total_medals','score_position'])

    st.write('Esporte')   
    st_show_dataframe_compacted(df.iloc[722],['sport_position','sport','sport_gold','sport_silver','sport_bronze','total_medals_sport'])

    st.write('Medalha') 
    st_show_dataframe_compacted(df.iloc[722],['modality','athlete','medal'])

    st.write('Decidiu-se, para facilitar a análise, manter a tabela na forma não-normalizada. Ou seja, para o Brasil, todas as linhas repetirão as informações atribuídas ao país. Da mesma forma, as informações sobre a Ginástica Artística do Brasil também serão replicadas em todas as linhas referentes às medalhas brasileiras desse esporte.')

    with open('config/country_names.json', 'r', encoding='utf-8') as file:
        country_rep = json.load(file)
    df['name'] = df['name'].map(lambda x:country_rep[x] if x in country_rep.keys() else x)


    st.divider()

    st.subheader('Categorias selecionadas.')

    st.write('Com os dados em mãos, foram determinados cinco critérios (passaremos a chamar também de _Features_) para definir os grupos (passaremos a chamar também de _Clusters_) de delegações, com base em semelhanças nos perfis de medalhas:')

    st.markdown("<ul>", unsafe_allow_html=True)
    st.markdown("<li> Total de Medalhas 🥇🥈🥉</li>", unsafe_allow_html=True)
    st.markdown("<li> Percentual de Medalhas de Ouro 🥇➗🥇🥈🥉</li>", unsafe_allow_html=True)
    st.markdown("<li> Percentual de Medalhas em Modalidades Femininas 👩➗👥</li>", unsafe_allow_html=True)
    st.markdown("<li> Percentual de Medalhas em Modalidades Coletivas 👥➗👤👥</li>", unsafe_allow_html=True)
    st.markdown("<li> Número de Medalhas em Diferentes Categorias 🏌🏻⛹🏾🤸🏾🏋🏻 </li>", unsafe_allow_html=True)
    st.markdown("</u>", unsafe_allow_html=True)


    st.write('Passamos, então, a direcionar nosso foco para a análise da tabela sob a perspectiva de cada país, realizando operações matemáticas para calcular os valores mencionados anteriormente.') 

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
 
    df_features = df_by_country.copy()
    df_features = df_features.reset_index().drop('index', axis=1).set_index('name') 
    df_features.index = df_features.index.map(name_symbol_dict)

    
    st.write('Em relação aos critérios de totalização, especificamente o primeiro e o último, realizaremos a normalização com base nos maiores valores observados em ambos.')

    st.write('Agora estamos prontos para começar a aplicar as técnicas de machine learning para identificar os perfis de países.')

    st.divider()

    st.title('Aplicação de Técnicas de Machine Learning')


    st.subheader('Escolha do número de grupos a serem formados')

    st.write('Para iniciarmos a identificação dos clusters, é essencial determinar a quantidade ideal. Um número muito baixo pode resultar em grupos heterogêneos e pouco representativos. Por outro lado, uma quantidade muito alta pode tornar a análise excessivamente específica e menos eficaz.')


    st.markdown('<p class="techinfo">✍️ Uma análise que pode nos ajudar nessa decisão é a observação do gráfico do tipo "cotovelo" (elbow). Esse gráfico indica a faixa em que o número ideal de clusters pode ser encontrado. Por meio de uma série de simulações, o gráfico nos mostra a variabilidade interna de acordo com o número de clusters, indicando o ponto em que adicionar mais clusters não resulta em melhorias significativas na separação dos dados. <br /><br />No gráfico abaixo, podemos observar exatamente isso. Inicialmente, a variabilidade interna é muito alta, mas a redução das divergências internas é significativa até o 7º cluster, tornando-se mínima a partir desse ponto.</p>', unsafe_allow_html=True)



    st.write('Com base no gráfico do cotovelo e na observação dos agrupamentos obtidos, definimos o uso de 8 grupos para esta análise.')

    elbow_chart(df_features,show)

    # Número de clusters escolhido (exemplo: 8)
    optimal_k = 8

    st.subheader('Aplicando do algoritmo de Clusterização e visualização dos grupos')

    st.write('Prossegue-se então para a realização da aplicação do algorítmo de clusterização: o KMeans.')

    st.markdown('<p class="techinfo">✍️ Este algoritmo opera da seguinte forma: é definido um ponto central para representar cada um dos oito grupos e atribui cada elemento (neste caso, países) ao grupo mais próximo. Em seguida, recalcula a posição do ponto central de cada grupo, ajustando-o para a posição média dos elementos atribuídos ao grupo. Após isso, é revisto o grupo a qual cada elemento pertence com base na novas posições dos centros. Esse processo é repetido até que essas posições não mudem mais.</p>', unsafe_allow_html=True)
    
    # Aplica o método de clusterização
    clusters,scaled_features  = apply_kmeans(optimal_k,df_features,name_symbol_dict)

    # Adicionar os clusters ao DataFrame de países
    df_by_country['Cluster'] = clusters
    df_by_country['Symbol'] = df_features.index

    st.write('Algoritmo aplicado, agora vamos ver como ficou!')

    st.write('Para a visualização dos grupos formados, um passo adicional é necessário, já que o algoritmo de clusterização gera resultados em dimensões superiores às que podemos enxergar. Portanto, precisamos representar os clusters em 2 dimensões (plano) ou 3 dimensões (espaço). Para isso, utilizamos a técnica chamada PCA (Principal component Analysis).')

    st.markdown('<p class="techinfo">✍️ O PCA é responsável por "achatar" um espaço multidimensional em dimensões menores, mantendo o máximo de variabilidade dos dados. Para alcançar isso, o PCA realiza uma transformação matemática baseada em uma combinação das variáveis originais. Esta transformação é feita através da soma ponderada dos valores de cada uma das características (features).</p>', unsafe_allow_html=True)
    
    st.write('Podemos afirmar que o PCA pode ser utilizado para substituir as features na representação dos dados, pois cada componente principal é uma combinação das variáveis originais. No gráfico abaixo, podemos ver a contribuição de cada feature para a formação de cada componente principal (PC1, PC2 e PC3). ')

    # Mostra o critério de relevância de feature por cluster
    show_feature_to_pca(scaled_features,feature_names,show)

    st.write('Vamos analisar como o PC1 (Primeiro Componente Principal) é formado a partir dos valores das features. Esse raciocínio pode ser estendido para os outros componentes principais. Conforme detalhado na tabela acima, temos que:')
    st.code('PC1=0.6*[Nº Medalhas]+0.2*[% de Ouros]+0.3*[% Feminino]-0.4["% Mod. Coletivas]+0.6[Nº Categorias]')

    st.write('Agora que explicamos a necessidade do uso do PCA e mostramos como ele é formado a partir dos valores das features, estamos prontos para representar os dados em um gráfico de duas dimensões, utilizando o PC1 e o PC2. Veja o resultado:')
    # Mostra os cluster em um espaço 2d
    print_23d_clusters(df_by_country,scaled_features,name_symbol_dict,labels,2,show)

    st.write('Visualizamos agora os oito clusters formados e seus respectivos integrantes com base em suas localizações nos eixos dos dois primeiros componentes principais (PC1 e PC2).') 

    st.write('A partir da dispersão apresentada, é possível perceber que alguns grupos, como os clusters 4, 5 e 8, podem ser facilmente separados dos demais. No entanto, a separação de outros grupos não é tão simples, pois eles estão emaranhados em uma região específica do gráfico.')

    st.write('Observando os países que compõem esse conjunto e lembrando que o PC1 (no eixo horizontal) é fortemente influenciado pelo número de medalhas, podemos concluir que esse conjunto de grupos é composto por países com poucas medalhas.')  

    st.write('Devido a essa dificuldade de separação, a representação em duas dimensões não é suficiente para definir claramente os limites desses clusters. Para uma visualização mais eficaz, precisamos adicionar pelo menos mais uma dimensão. Vamos então explorar o espaço tridimensional.') 


    st.write('Ao adicionar o terceiro componente (PC3), conseguimos realizar a representação espacial dos dados. Assim, podemos explorar melhor a disposição dos clusters e países, e a separação entre eles torna-se muito mais evidente.')

    st.markdown('<p class="alert">⚠️ Para explorar o gráfico abaixo, navegue por ele utilizando o botão esquerdo do mouse para girar e o scroll para dar zoom ou recolher, ou então, use os controles fornecidos', unsafe_allow_html=True) 
    

    # Mostra os cluster em um espaço 3d
    print_23d_clusters(df_by_country,scaled_features,name_symbol_dict,labels,3,show)
  

    st.markdown('<p class="techinfo">✍️ A maior capacidade de separar visualmente os grupos ocorre porque, como mostrado na tabela acima, a adição de uma terceira dimensão aumenta a capacidade do PCA de representar a variabilidade dos dados. Esse índice é conhecido como variância explicada cumulativa.</p>', unsafe_allow_html=True) 

    st.write('Percebeu que foram adicionados nomes aos clusteres, vou explicar a partir do gráfico abaixo como chegamos nessa nomenclatura.') 

    st.divider()

    st.subheader('Atribuição de nomes aos grupos encontrados') 

    st.write('O gráfico do tipo mapa de calor (heatmap) utiliza cores para representar os valores, onde cores mais quentes indicam valores mais altos. No nosso caso, estamos analisando a média de cada feature por cluster em comparação com a média global de cada feature.') 
        
    # Mostra o heatmat de relevância de feature por cluster
    create_cluster_feature_heatmap(df_features,feature_names,df_by_country,labels,show)
     

    st.write('Por meio de inspeção visual, podemos analisar quais são as features mais marcantes em um cluster, tanto do ponto de vista de valores baixos (próximo de 0), quanto no caso de valores altos (maior que 1).')

    st.write('De imediato, é visível que o cluster 4 se refere às "Superpotências Olímpicas", devido ao alto número de medalhas e à diversidade de categorias esportivas nas quais as delegações integrantes conquistaram medalhas. O processo de definição de nomes prossegue para os demais grupos sempre com base em suas características marcantes.')

    # Compatibilização do símbolo para gerar o mapa
    df_sym_cluster,dict_country_label = map_symbols_to_clusters(df_by_country,labels,name_symbol_dict)
    world = fill_groups(df_sym_cluster)

    st.divider()

    st.title('Criação do Mapa colorido pelos grupos')
    
    st.write('O passo final é gerar o mapa colorido pelas cores dos grupos de países identificados com base nas suas características olimpicas. Veja como ficou!')
    
    generate_map(world,optimal_k,labels,show)

    
    st.markdown("Abaixo estão detalhados os países existentes em cada grupo:<ul>", unsafe_allow_html=True)
    for i in range(len(dict_country_label)):
        label = list(dict_country_label.keys())[i]
        st.markdown("<li  class='label_"+str(i)+"'><b>"+label+": </b><span class='country'>"+"; ".join(dict_country_label[label])+".<span></li>", unsafe_allow_html=True)
    st.markdown("</ul>", unsafe_allow_html=True)    

    st.markdown('<small>Obs. O grupo "Conquistas Femininas no Caribe" não está representado no gráfico por conta da dimensão das ilhas.</small>', unsafe_allow_html=True)
 
    st.write('Chegamos ao final deste projeto, demonstrando que é possível realizar uma análise diferenciada a partir do quadro de medalhas. A pergunta que fica é se, na próxima edição das Olimpíadas, essas características vão se manter ou teremos grandes mudanças.') 

    st.write('Será que finalmente veremos o Brasil ser incluído no grupo dos "Supermedalhistas Multimodais"?')

    st.link_button('Veja o código no github', 'https://github.com/gabrielmrp/olympic-country-groups', type="secondary")


    # TODO remover os eixos sem precisar cortar as imagens
    #crop_image(show)
