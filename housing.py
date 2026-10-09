# %%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


# %%

housing = pd.read_csv("Data/housing.csv")
housing.head()

# %%

housing.info()

# rápida descrição dos dados
# atributos (características), tipo (float, object), total 
# de linhas e nulos   

# %%

housing["ocean_proximity"].value_counts()

# determina quantas categorias ocean_proximity possui

# %%

housing.describe()

# apresenta as estatísticas das variáveis

# %%

housing.hist(bins=50, figsize=(20,15))
plot.show()


# %%
def split_train_test(data, test_ratio):
    # Definir o seed garante que o "aleatório" seja sempre reprodutível
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

# %%
train_set, test_set = split_train_test(housing, 0.2)
len(train_set)

# %%
len(test_set)

# %%
# divisão do dataset em treino e teste usando 
# o scikit-learn

train_set, test_set = train_test_split(housing, test_size=0.2, random_state= 42)
# %%

# income_cat é o salário em faixas (categorias)

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins= [0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels = [1,2,3,4,5])
# %%
housing["income_cat"].hist()

# %%
# Amostragem estratificada com base na categoria de renda

split = StratifiedShuffleSplit(n_splits=1, test_size= 0.2, random_state= 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_test_set["income_cat"].value_counts() / len(strat_test_set)
# %%

# remover o atributo income_cat para que os dados voltem ao normal

for set_ in (strat_train_set, strat_test_set):
    set_ .drop("income_cat", axis= 1, inplace=True)
# %%
