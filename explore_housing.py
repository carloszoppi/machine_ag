# %%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# %%
housing = pd.read_csv("Data/housing.csv", sep=",")
housing.head()

# %%
# divisão do dataset em treino e teste usando 

train_set, test_set = train_test_split(housing, test_size=0.2, random_state= 42)

# %%

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins= [0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels = [1,2,3,4,5])

housing["income_cat"].hist()


# Amostragem estratificada com base na categoria de renda

split = StratifiedShuffleSplit(n_splits=1, test_size= 0.2, random_state= 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# %%
strat_train_set.shape

# %%

# Vamos trabalhar com uma cópia do data set de treino:

housing = strat_train_set.copy


# %%

housing.plot(kind = "scatter", x = "longitude", y = "latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar = True,
             )
plt.legend()



# %%
housing_num = housing.select_dtypes(include='number')
corr_matrix = housing_num.corr()
corr_matrix["median_house_value"].sort_values(ascending = False)



# %%
