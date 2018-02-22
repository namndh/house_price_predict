import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from unreleased_lib.CategoricalEncoder import CategoricalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

PROJECT_PATH = '/home/t3min4l/workspace/house_price_predict/main/'
FIGURE_PATH = os.path.join(PROJECT_PATH,'figures')

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombineAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_name):
        self.attributes_name = attributes_name
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        return X[self.attributes_name].values
    

def load_housing_data(data_path=PROJECT_PATH + 'data'):
    csv_path = os.path.join(data_path, 'housing.csv')
    return pd.read_csv(csv_path)

def save_fig(fig_id, tight_layout = True, fig_extension = 'png', resolution=300):
    path = os.path.join(FIGURE_PATH, fig_id + '.' + fig_extension)
    print('Saving figure', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

housing = load_housing_data()
# print(housing.head())
# print(housing.info())
# print(housing['ocean_proximity'].value_counts())
# print(housing.describe())
# housing.hist(bins=50, figsize=(30,25))
# plt.show()
# housing['median_income'].hist()
# plt.show()
# split data into train set and test set according to income category using stratification to get a representative sets of the data.
housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
for set_ in (strat_train_set, strat_test_set):
    set_.drop('income_cat', axis=1, inplace=True)

# Vizualization of the data
housing = strat_train_set.copy()

def plot_cali_house_price():
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
    save_fig('california_house_price_plot')
    plt.show()

# plot_cali_house_price()


# Correlation of the features in data
corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))
# housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.4)
# plt.show()


# Combine Attributes
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_rooms'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']

corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending=False))


housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

housing_numerical = housing.drop('ocean_proximity', axis=1)

num_attribs = list(housing_numerical)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy='median')),
    ('attrib_adder', CombineAttributesAdder()),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', CategoricalEncoder(encoding="onehot-dense"))
])

full_pipeline = FeatureUnion(transformer_list = [
    ('num_pipeline', num_pipeline),
    ('cat_pipline', cat_pipeline)
])

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing_prepared.shape)