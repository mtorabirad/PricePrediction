## Description of the code structure

The notebook contains an object-oriented code, developed using scikit-learn, that builds linear and tree-based regression models. The code consists of the following seven classes:

<ins>CleanData</ins>: contains all the methods responsible for cleaning data, such as: replacing symbols, standardizing letters, and converting data types.
<ins>ExploreData</ins>: contains methods responsible for exploring data such as printing summary information.
PreprocessData</ins>: contains methods responsible for preprocessing data, such as encoding/scaling catagorical/numerical variables 

<ins>EDA</ins>: This is the class that performs the actual explatory data analysis and it inherits the methods implemented in the above classes. 

<ins>SelectFeatures</ins>: An "abstraact" class that implements recursive feature elimination. 
<ins>BuildModelAfterSelectingFeatures</ins>: This class inherits from the SelectFeatures class, performs the actual feature selection, builds the model, performs hyper-parameter tunning, and makes predictions.

<ins>EvaluateTheModel</ins>: this class implements all the methods responsible for evaluating different models, such as calculating RMSE and r2 scores and methods that check whether the underlying assumptions of linear regress are satisfied. 

## Instructions to continue

If you would like to run the notebook, first download the csv file (containing Airbnb listings.csv) and then set the variable "CSV_local_address" in the second cell of the notebook to the local addrress of the downloaded file.

## Requirements
Python 3.6 <br/>
Scikit-learn<br/>
Pandas<br/>
Numpy<br/>
seaborn<br/>
scipy<br/>
statsmodels<br/>