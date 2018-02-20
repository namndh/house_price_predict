import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.model_selection import StratifiedShuffleSplit

PROJECT_PATH = '/home/t3min4l/workspace/house_price_predict/main/'


def load_housing_data(data_path=PROJECT_PATH + 'data'):
    csv_path = os.path.join(data_path, 'housing.csv')
    return pd.read_csv(csv_path)


housing = load_housing_data()
# print(housing.head())
# print(housing.info())
# print(housing['ocean_proximity'].value_counts())
# print(housing.describe())
# housing.hist(bins=50, figsize=(30,25))
# plt.show()
# housing['median_income'].hist()
# plt.show()
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)


housing = strat_train_set.copy()

cali_img = imread(PROJECT_PATH + 'images/california.png')
ax = housing.plot(kind='scatter', x='longitude', y='latitude', figsize=(10,7),
                    s=housing['population']/100, label='Population',
                    c='median_house_value', cmap=plt.get_cmap('jet'),
                    colorbar=False, alpha=0.4)
plt.imshow(cali_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
plt.xlabel('longitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)

prices = housing['median_house_value']
tick_values = np.linspace(prices.min(), prices.max(), 10)
cbar = plt.colorbar()
cbar.ax.set_yticklabels(['$%dk'%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
plt.show()

corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))