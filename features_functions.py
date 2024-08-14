import pandas as pd


def def_gender(x):
    x = x.lower()
    if 'masculin' in x or 'boy' in x :
        return 'masculino'
    if 'feminin' in x or 'girl' in x :
        return 'feminino'
    if 'misto' in x:
        return 'misto'
    return None

def get_feature_total_medals(df_by_country):
    df_by_country['total_medals_norm'] = df_by_country['total_medals'].apply( lambda x: x / df_by_country['total_medals'].max())
    return df_by_country


def get_feature_gold_medals(df_by_country):
    df_by_country['gold_medals_perc'] = df_by_country['gold'] / df_by_country['total_medals']
    return df_by_country

## calcula os casos em que a feature é definida pela divisão de duas colunas

def calculate_proportion(df, name_col, category_col,options, new_col_name):
    aux_df = pd.get_dummies(df.groupby([name_col,category_col]).count()['symbol'].reset_index(), columns=[category_col])

    sel_options = options[:2]

    for option in sel_options:
        aux_df[f'{category_col}_{option}'] = aux_df['symbol'] * aux_df[f'{category_col}_{option}'].astype(int)

    # Trata a especificidade do drop
    if len(options) == 3:
        aux_df.drop(['symbol',f'{category_col}_{options[2]}'], axis=1, inplace=True)
    else:
        aux_df.drop(['symbol'], axis=1, inplace=True)

    aux_df = aux_df.groupby(name_col).sum().reset_index()

    # Realiza a divisão
    aux_df[new_col_name] = aux_df[f'{category_col}_{options[0]}']  \
    	/ (aux_df[f'{category_col}_{options[1]}'] + aux_df[f'{category_col}_{options[0]}'])


    return aux_df
## gênero
def get_feature_gender(df,df_by_country):
    aux_df = calculate_proportion(df, 'name', 'gender',['feminino','masculino'],'most_fem')
    df_by_country = df_by_country.join(aux_df[['name','most_fem']].set_index('name'), on='name')

    return df_by_country

## coletivo ou individual

def get_feature_collective(df,df_by_country):
    aux_df = calculate_proportion(df, 'name', 'collective',['individual','team'],'most_individual')
    df_by_country = df_by_country.join(aux_df[['name','most_individual']].set_index('name'), on='name')

    return df_by_country

## número de categorias diferentes

def get_feature_categories_diverse(df,df_by_country):

    aux_df = df[['categories']].value_counts().reset_index()
    medals_count = df[['categories']].value_counts().sum()

    aux_df['count'] = aux_df['count'].apply( lambda x: x / medals_count)

    aux_df = pd.get_dummies(df.groupby(['name','categories']).count()['symbol'].reset_index(), columns=['categories'])
    df_by_country['categories_diverse'] = df_by_country['name'].map((aux_df['name'].value_counts() \
    	 / len(df[['categories']].drop_duplicates())).to_dict())

    return df_by_country

#FEATURES
