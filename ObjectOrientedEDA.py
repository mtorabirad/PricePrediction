import pandas as pd
import numpy as np
import os
import random
from matplotlib import pyplot as plt

from category_encoders import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns; 

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from lightgbm import LGBMRegressor
from yellowbrick.model_selection import RFECV
from sklearn.feature_selection import RFE

#from geopy.distance import great_circle
#from math import nan # You can use None without importing anything.


random.seed(1234)

""" 
Naming conventions to be followed:
Class names: singular and capitalized (e.g.,  'Book' rather than 'book' or 'Books')
Member functions: lowercase verbs and separeted by underscore for multple words (e.g., preprocess_data)
Variables: lowercase nouns and separeted by underscore for multple words (e.g., product_id)
Constant: uppercase  nouns and separeted by underscore for multple words (e.g., MAX_SIZE) 
"""

""" Next: for exception handling use the following pattern

try: 
    write_to_file(f) 
except: 
    print('Failed') 
else: 
    print('Succeeded')
finally: 
    f.close() """

def read_df_select_rand_rows(address):
    # Randomly select only n% of the rows and skip the rest of the rows. Make sure to keep the header row.
    f = address
    num_lines = sum(1 for l in open(f, encoding='utf-8'))
    n = 99
    size = int(n*0.01*num_lines)
    skip_idx = random.sample(range(1, num_lines), num_lines - size) # range form 1 to make sure header row is not skipped.
    selectedf = pd.read_csv(f, skiprows=skip_idx)
    #selectedf = pd.read_csv(f)
    #selectedf = pd.read_csv(f, nrows=10)
    return selectedf

def distance_to_mid(lat, lon):
    #berlin_centre = (52.5027778, 13.404166666666667)
    Toronto_center = (43.653225, -79.383186)
    accommodation = (lat, lon)
    return great_circle(Toronto_center, accommodation).km

class CleanData():

    def __init__(self):                
        """
        This class will contain ALL the functions that may be used to CLEAN data. At the moment, these include:
            replace_these_symbols_in_these_cols(replace_dic)
            standardize_letters_in_cat_cols()
            convert_the_type_of_these_cols(dic)
            drop_these_cols(dic)
        """
        print("CleanData object created")
    
    def replace_these_symbols_in_these_cols(self, replace_dic):
        self._replace_these_symbols_in_these_cols(replace_dic)  

    def standardize_letters_in_cat_cols(self):
        self._standardize_letters_in_cat_cols()      
    
    def convert_the_type_of_these_cols(self, dic):
        self._convert_the_type_of_these_cols(dic)

    def drop_these_cols(self, cols):
        self._drop_these_cols(cols)

    def handleNaN(self):
        self._handleNaN()
    
    # Private methods (abstracted out)
    def _replace_these_symbols_in_these_cols(self, replace_dic):        
        for col, symbols in replace_dic.items():
            try:        
                self.data[col] = self.data[col].str.replace(symbols[0], symbols[1])
            except:
                print('In col {}, {} cannot be replaced with {}'.format(col, symbols[0], symbols[1]))
               
    def _convert_the_type_of_these_cols(self, dic):
        for col, cast_to_type in dic.items():
            self.data[col] = self.data[col].astype(cast_to_type)

    def _standardize_letters_in_cat_cols(self):
        num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'uint8']
        for column in self.data.columns:
            if self.data[column].dtypes not in num_types:
                #print(self.data[column].dtypes)
                self.data[column] = self.data[column].str.lower()

    def _drop_these_cols(self, cols):
        for column in cols:
            self.data=self.data.drop(labels=column, axis=1)
    
    def _handleNaN(self):
        #for col in cols:
        #    #self.data = self.data[pd.notnull(self.data[col])] 
        self.data = self.data.dropna(axis=0,how='all')
        num_cols = self.data.select_dtypes(exclude='object').columns
        cat_cols = self.data.select_dtypes(include='object').columns
        self.data[cat_cols] = self.data[cat_cols].apply(lambda col: col.fillna(col.mode()[0]))
        self.data[num_cols] = self.data[num_cols].apply(lambda col: col.fillna(col.median()))
######################
class ExploreData():

    def __init__(self):
        """
        This class will contain ALL the functions that may be used to explore data. At the moment, these include:
            print_summary_info()
            plot_histogram(cols)
        
        Other useful EDA methods: 
        df.describe() is useful for summarizing numeric data while df.value_counts() are good for categorical data.
        data.info()
        data.isnull().sum()
        sns.boxplot(x=data['Horsepower'])
        Before decising how to impute missing values, you should look at box plots. 
        For example, if the plot shows outliers, you should impute with median, not mean. For imputing, use 
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy="median")imputer.fit(num_data)
        
        To look for the category distribution in categorical columns us data["CatCol"].value_counts()
        """
        print("Information object created")

    def print_summary_info(self, detailed=0):
        """
        print feature name, data type, number of missing values and ten samples of 
        each feature
        :param data: dataset information will be gathered from
        :return: no return value
        """
        
        
        if detailed == 0:
            print("The dataset now has {} rows and {} columns .".format(*self.data.shape))
            #print(self.data.info(verbose=False))           
            print(self.data.columns.values)
        
        if detailed != 0:
            
            feature_dtypes=self.data.dtypes
            self.number_of_missing_values_for_each_col=self._get_number_of_missing_values_for_each_col(self.data)       
            # In the line below, instead of {:45} use the maximum # of characters in the feature names.
            print("{:45} {:21} {:21} {:21}".format("Feature Name".upper(), "Data Format".upper(), "# of Missing Values".upper(), "Samples".upper()))
            for feature_name, dtype, missing_value in zip(self.number_of_missing_values_for_each_col.index.values,
                                                        feature_dtypes[self.number_of_missing_values_for_each_col.index.values],
                                                        self.number_of_missing_values_for_each_col.values):
                print("{:48} {:21} {:21} ".format(feature_name, str(dtype), str(missing_value)), end="")
                if detailed != 0:
                    for v in self.data[feature_name].values[:3]:
                        print(v, end=",")
                print()            
        print("=" * 150)
    
    def plot_histogram(self, cols):
        
        if not os.path.exists(os.getcwd() + '/Figures'):
            os.makedirs(os.getcwd() + '/Figures')

        for col in cols:
            print('col=', col)
            _, axs = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    
            axs.hist(self.data[col])
            axs.set_xlabel('binned values')
            axs.set_ylabel('frequency')
            address_and_name = os.getcwd() + '/Figures/Histogram' + str(col)
            plt.savefig(address_and_name)
    
    def _get_number_of_missing_values_for_each_col(self,data):
        """
        Find missing values of given datad
        :param data: checked its missing value
        :return: Pandas Series object
        """
        #Getting sum of missing values for each feature
        missing_values = data.isnull().sum()
        print('missing_values=', missing_values)
        #Feature missing values are sorted from few to many
        
        missing_values.sort_values(ascending=False, inplace=True)
        
        #Returning missing values
        return missing_values
######################
class PreprocessData():    
    def __init__(self, data):
        """
        This class will contain ALL the functions that may be used to preprocess data. At the moment, these include:
            encode_then_delete_these_cat_cols(cols)
            scale_these_cols(cols)

        """
        print("FeatureEngineering object created")
    
    def encode_then_delete_these_cat_cols(self, cols):
        self._encode_then_delete_these_cat_cols(cols)

    def scale_these_cols(self, cols, method):
        self._scale_these_cols(cols, method)

    def _scale_these_cols(self, cols, method):
        #self.data['price'] = MinMaxScaler().fit_transform(self.data[['price']]) #Two brackets because MinMaxScalar can accept a pandas dataframe but not a series
        '''Function to scale the features
        Normalization: 
            Subtract the min, and divide by max - min, such that all the features end up ranging between 0 and 1.
            Recommended when the distribution is NOT Gaussian. 

        Standardization: S
            Subtract the mean and divide by standard deviation, such that the mean and std 
            of the scaled features are 0 and 1, respectivelly.
            Recommended when the distribution is Gaussian. 

        Input:
            df: df to be scaled - all the columns should be integer or float values. 
        '''        
        for col in cols:
            self.data[col] = MinMaxScaler().fit_transform(self.data[[col]])
        
    def _encode_then_delete_these_cat_cols(self, cols):
        """ Use binary encoding to encode two categorical features
        For a list of encoders: https://contrib.scikit-learn.org/category_encoders/

        OrdinalEncoder: Encodes categorical features as ordinal, in one ordered feature.
        OneHotEncoder: Onehot (or dummy) coding for categorical features, produces one feature per category, each binary.
        BinaryEncoder: Binary encoding for categorical variables, similar to onehot, but stores categories as binary bitstrings. """
        
        # Problem: How to assign Labels with Sklearn category_encoders?
        # Right now, only OneHotEncoder can include the category values in the encoded column names (use_cat_names=True)
        # In the link below, OneHotEncoder of sklearn.preprocessing (not category_encoders) 
        # is extended to allow using catagorical variables to label the columns
        # https://towardsdatascience.com/how-to-assign-labels-with-sklearn-one-hot-encoder-e59a5f17df4f
        # Is it possible to do a similar thing on the OneHotEncoder of the category_encoders?

        #encoder = OrdinalEncoder()
        #encoder = OneHotEncoder(use_cat_names=True) 
        #return pd.concat([df, encoder.fit_transform(df[column])],axis=1)
        for col in cols:
            self.data = pd.concat([self.data, pd.get_dummies(self.data[col], prefix=col)],axis=1)
            self.data = self.data.drop(columns = col)
        # The next three lines should not be in this method. They are not related to encoding. 
        #numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'uint8']
        #self.data = self.data.select_dtypes(include=numerics)
        #self.data = self.data.dropna(axis=1, how='all')
######################
class ObjectOrientedEDA(ExploreData, CleanData, PreprocessData):
    """
        This class will cleate the data object and invoke the methods implemented in classes ExploreData, CleanData, and PreprocessData.
        The methods of this class ONLY can be invoked from the main. 

    """
    def __init__(self, data1, data2=None):        
        self.train=data1
        self.test=data2
        self.data=self._concat_two_datasets()
        #print("ObjectOrientedTitanic object created")

    def explore_data_print_summary_info(self, detailed=0):             
        super(ObjectOrientedEDA, self).print_summary_info(detailed)
    
    def plot_histogram_of_this_column(self, col):
        super(ObjectOrientedEDA, self).plot_histogram([col])
    
    def clean_data(self):
        # Drop any col that has url in it
        cols_with_url = []
        for col in self.data.columns:
            if 'url' in col:
                cols_with_url.append(col)
        #print(cols_with_url)
        super(ObjectOrientedEDA, self).drop_these_cols(cols_with_url + ['host_name', 'host_id', 
                                                        'id', 'name', 'neighbourhood', 'last_review', 
                                                        'calculated_host_listings_count', 'reviews_per_month', 
                                                        'minimum_nights', 'availability_365', 'number_of_reviews', 'city', 'state', 'country_code',
                                                        'country'])
        
        # Define a replacement dictionary like this: {'$':'', ',':'', ' ':''} and a colum list
        # Pass list of cols and the replacement dictionary
        #['price', 'weekly_price', 'monthly_price', 'extra_people', 'security_deposit', 'cleaning_fee']
        # Perhaps, something similar to this: self.data.host_is_superhost = self.data.host_is_superhost.map(dict(t=1, f=0))
        # Also, think about taking advantage of the fact that functions are object so you can make 
        # a list of the operations you want to apply and then invoke them in a for loop by "for function in function_operations":
        # For more details, see page 72 of Python for Data Analysis (second edition)
        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'price': ['$', '']})
        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'price': [',', '']})
        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'price': [' ', '']})
        # Will using a composite ditionary help here? directory[last,first] = number

        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'weekly_price': ['$', '']})
        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'weekly_price': [',', '']})
        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'weekly_price': [' ', '']})

        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'monthly_price': ['$', '']})
        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'monthly_price': [',', '']})
        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'monthly_price': [' ', '']})

        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'extra_people': ['$', '']})
        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'extra_people': [',', '']})
        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'extra_people': [' ', '']})

        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'security_deposit': ['$', '']})
        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'security_deposit': [',', '']})
        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'security_deposit': [' ', '']})

        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'cleaning_fee': ['$', '']})
        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'cleaning_fee': [',', '']})
        super(ObjectOrientedEDA, self).replace_these_symbols_in_these_cols({'cleaning_fee': [' ', '']})

        super(ObjectOrientedEDA, self).convert_the_type_of_these_cols({'price': 'float32'})
        self.data = self.data[self.data['price'] < 1000] # This really helps with the r2 score! 
        super(ObjectOrientedEDA, self).convert_the_type_of_these_cols({'cleaning_fee': 'float32'})
        super(ObjectOrientedEDA, self).convert_the_type_of_these_cols({'extra_people': 'float32'})
        super(ObjectOrientedEDA, self).convert_the_type_of_these_cols({'security_deposit': 'float32'})
        
        super(ObjectOrientedEDA, self).standardize_letters_in_cat_cols()
        super(ObjectOrientedEDA, self).handleNaN()

        # Drop all the cols except:
        selected_features  = ['cancellation_policy', 'is_location_exact', 'accommodates', 'property_type', 'price', 
        'host_total_listings_count', 'extra_people', 'guests_included', 'review_scores_rating', 'host_response_time',
        'host_is_superhost', 'room_type', 'instant_bookable', 'host_identity_verified', 'host_has_profile_pic',
        'security_deposit', 'host_listings_count', 'requires_license', 'neighbourhood_group_cleansed', 'neighbourhood_cleansed',
        'bathrooms', 'require_guest_phone_verification', 'cleaning_fee']
        
        selected_features  = ['price', 'room_type', 'accommodates', 'bathrooms', 'cleaning_fee', 
                            'cancellation_policy', 'property_type', 'extra_people']
        self.data = self.data[selected_features]

    def preprocess_data(self):
        super(ObjectOrientedEDA, self).encode_then_delete_these_cat_cols(['room_type'])
        super(ObjectOrientedEDA, self).encode_then_delete_these_cat_cols(['cancellation_policy'])
        super(ObjectOrientedEDA, self).encode_then_delete_these_cat_cols(['property_type'])

        """super(ObjectOrientedEDA, self).encode_then_delete_these_cat_cols(['is_location_exact'])
        super(ObjectOrientedEDA, self).encode_then_delete_these_cat_cols(['host_response_time'])
        super(ObjectOrientedEDA, self).encode_then_delete_these_cat_cols(['neighbourhood_cleansed'])
        super(ObjectOrientedEDA, self).encode_then_delete_these_cat_cols(['host_is_superhost'])
        super(ObjectOrientedEDA, self).encode_then_delete_these_cat_cols(['host_identity_verified'])
        super(ObjectOrientedEDA, self).encode_then_delete_these_cat_cols(['host_has_profile_pic'])
        super(ObjectOrientedEDA, self).encode_then_delete_these_cat_cols(['requires_license'])
        super(ObjectOrientedEDA, self).encode_then_delete_these_cat_cols(['require_guest_phone_verification'])
        super(ObjectOrientedEDA, self).encode_then_delete_these_cat_cols(['instant_bookable']) """

        #self.data.host_is_superhost = self.data.host_is_superhost.map(dict(t=1, f=0))
        
        super(ObjectOrientedEDA, self).scale_these_cols(['price'], 'Normalization')
        super(ObjectOrientedEDA, self).scale_these_cols(['accommodates'], 'Normalization')
        super(ObjectOrientedEDA, self).scale_these_cols(['bathrooms'], 'Normalization')
        super(ObjectOrientedEDA, self).scale_these_cols(['cleaning_fee'], 'Normalization')
        super(ObjectOrientedEDA, self).scale_these_cols(['extra_people'], 'Normalization')
        
    def _concat_two_datasets(self):
        return pd.concat([self.train, self.test])
######################
class FeatureSelection():
    def __init__(self, data):
        self.data = data  
        self.model = DecisionTreeRegressor()
        #self.model = LGBMRegressor()
        #self.model = xgb.XGBRegressor()      
        self.target = data[["price"]]
        self.selected_features = data.drop(["price"], axis=1)
        self.num_features_to_select = 5
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.selected_features, self.target, test_size=0.2, random_state=1)
        self.selected_features_name = None
        self._recursive_feature_elimination()
        self._drop_non_selected_features_and_split()
        
    def _recursive_feature_elimination(self):
        """ visualizer = RFECV(self.model, verbose=1, n_jobs = 4)
        visualizer.fit(self.X_train, self.y_train)
        visualizer.show() """

        rfe = RFE(self.model, self.num_features_to_select, verbose=0)

        rfe = rfe.fit(self.X_train, self.y_train)        
        feature_index_list = []
        features_list = []
        for feature_index, TrueFalse in enumerate(rfe.get_support(), start=0):
            if TrueFalse == True:
                feature_index_list.append(str(feature_index))
        
        for feature_index, i in enumerate(self.X_train.columns.values, start=0):
            if str(feature_index) in feature_index_list:
                features_list.append(self.X_train.columns.values[feature_index])

        self.selected_features_name = features_list
        print(f"selected features: {features_list}")
    def _drop_non_selected_features_and_split(self):
        for col_name in self.data.columns.values:
            if col_name not in self.selected_features_name:
                #print(f"{col_name} is not selected so it sill be dropped.")
                pass
        self.selected_features = self.selected_features[self.selected_features_name]
        
        # Re-split, this time with only the selected features.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.selected_features, self.target, test_size=0.2, random_state=1)
    
class FitandPredict(FeatureSelection):
    def __init__(self, data):
        super(FitandPredict, self).__init__(data)
        self.y_train_pred = None
        self.y_test_pred = None
        self.RMSE_test = None
        self.r2_test = None
        self.RMSE_train = None
        self.r2_train = None
        self.feat_imp = None
        
        print("FitandPredict object created")

    def tune_and_reset_hyperparameters(self):
        
        """ param_grid = {'n_estimators': [100, 150, 200],
              'learning_rate': [0.01, 0.05, 0.1], 
              'max_depth': [3, 4, 5, 6, 7],
              'colsample_bytree': [0.6, 0.7, 1],
              'gamma': [0.0, 0.1, 0.2]} """

        param_grid = {  'num_leaves' : [21, 31, 41],
                        'n_estimators': [50, 70, 90],
                        'learning_rate': [0.05, 0.07, 0.1]
                     }
        # instantiate the tuned random forest
        model_grid_search = GridSearchCV(self.model, param_grid, cv=3, n_jobs=-1)

        # train the tuned random forest
        model_grid_search.fit(self.X_train, self.y_train)

        #print(type(model_grid_search.best_params_))
        print('model_grid_search=', model_grid_search.best_params_)

        dic = model_grid_search.best_params_
        # manual intervention: instantiate xgboost with best parameters
        self.model =  LGBMRegressor(num_leaves = dic['num_leaves'], n_estimators = dic['n_estimators'], learning_rate = dic['learning_rate'])
        #self.model =  xgb.XGBRegressor(n_estimators = dic['n_estimators'], learning_rate = dic['learning_rate'])

    def fit_and_predict(self):

        self.model.fit(self.X_train, self.y_train) 
        self.y_train_pred = self.model.predict(self.X_train)
        self.y_test_pred = self.model.predict(self.X_test)
        self._print_errors()

    def _print_errors(self):

        self.RMSE_test = np.sqrt(mean_squared_error(self.y_test, self.y_test_pred))
        self.r2_test = r2_score(self.y_test, self.y_test_pred) # https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score

        print(f"RMSE_test: {round(self.RMSE_test, 4)}")
        print(f"r2_test: {round(self.r2_test, 4)}")

        self.RMSE_train = np.sqrt(mean_squared_error(self.y_train, self.y_train_pred))
        self.r2_train = r2_score(self.y_train, self.y_train_pred)

        print(f"RMSE_train: {round(self.RMSE_train, 4)}")
        print(f"r2_train: {round(self.r2_train, 4)}")

        print(f"r2_test/r2_train {round(self.r2_test, 4)/round(self.r2_train, 4)}")

        #self.feat_imp = pd.Series(self.model.feature_importances_, index=objectOrientedEDA.data.columns.drop('price'))

        #self.feat_imp.nlargest(15).plot(kind='barh', figsize=(10, 6))
        #plt.xlabel('Relative Importance')
        #plt.title("Feature importances", fontsize=18, fontweight='bold')
        #plt.show()
        ###########
######################

if __name__ == "__main__":

    df_address = r"C:\DATA\Projects\PortFolioProjects\September2020\Tor\DetailedData\listings.csv"
    #C:\DATA\Projects\PortFolioProjects\September2020\Tor\DetailedData
    working_df = read_df_select_rand_rows(df_address)
    
    objectOrientedEDA=ObjectOrientedEDA(working_df)
   
    objectOrientedEDA.data.to_csv('Original'+'.csv', index=False)
    
    objectOrientedEDA.explore_data_print_summary_info(1)
    objectOrientedEDA.plot_histogram_of_this_column('price')
    
    objectOrientedEDA.clean_data()
    #objectOrientedEDA.explore_data_print_summary_info(1)
    corr_matrix = abs(objectOrientedEDA.data.corr())
    pd_series = corr_matrix['price'].sort_values(ascending=False)
    print(pd_series[abs(pd_series) > 0.1])
    

    objectOrientedEDA.preprocess_data()
    #objectOrientedEDA.explore_data_print_summary_info(1)
    
    #objectOrientedEDA.data['distance'] = objectOrientedEDA.data.apply(lambda x: distance_to_mid(x.latitude, x.longitude), axis=1)

    
    #objectOrientedEDA.data.to_csv('Final'+'.csv', index=False)
    print(objectOrientedEDA.data.info)
    
    FitandPredictObject = FitandPredict(objectOrientedEDA.data)
    
    #FitandPredictObject.tune_and_reset_hyperparameters()
   
    FitandPredictObject.fit_and_predict()
    exit("MTR Exit")
   