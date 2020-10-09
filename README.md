# Predicting Airbnb prices using different Machine Learning algorithms

## Summary

In this project, I have shown how to build different Machine Learning models to predict Airbnb prices. One of the motivations of the project is the following business question: "What dictates Airbnb rental price? The type of the property? The number of people it can accommodate? Distance from the center? Review score? Cancellation policy?" 

After cleaning/preprocessing the data, Exploratory Data Analysis (EDA) was performed to identify outliers and possible inter-correlation between the features. Linear Regression (LR), Decision Tree (DT), Random Forest (RF), and boosting methods Xtreme Gradient Boosting (XGBoost) and Light Gradient Boosting (LGBoost) were trained. For LR, predictions were examined in detail and statistical hypothesis tests were performed to check whether the underlying assumptions of the model are satisfied or not. The key hyper-parameter controlling the accuracy of the DT and the ensemble methods were identified and tunned to maximize the R_squared test score. 

## Instructions to continue

If you would like to run the notebook, first download the csv file (containing Airbnb listings.csv) and the set the variable "CSV_local_address" in the second cell of the notebook to the local addrress of the downloaded file.

You can find a [detailed description of the project] (#Detailed Description) 

## Requirements
Python 3.6 <br/>
Scikit-learn<br/>
Pandas<br/>
Numpy<br/>
seaborn<br/>
scipy<br/>
statsmodels<br/>

## Detailed Description

The training dataset contains more than one-hundred features and twenty-thousand observations. However, most of the features cannot possibly have any predictive value (for example, those containing URLs). These features are dropped first and then Exploratory Data Analysis (EDA) is performed. In EDA, outliers are detected and handled using established statistical procedures. The presence of inter-correlations between the features are examined, confounded features are eliminated in favor of the confounder, and domain knowledge is used to engineer a new feature. 

Among the categorical features, those that contain a natural order are encoded using ordinal encoding, and the rest using pandas get_dummies. Because the data set was low-cardinality, there was no need to use more advanced encoding methods. Numerical features are scaled using the normalization method. Other scaling methods were also explored, but their influence on the results was found to be negligible. At the end of EDA, a hold-out set is prepared and set aside. 

Different Ml models were explored. The first model is the linear regression (LR). The predictions of LR are analyzed in detail to check whether they satisfy the four underlying assumptions of LR: errors should have 1) zero mean and 2) constant variance (homoskedastic) and be 3) uncorrelated and 4) normally distributed. The analysis shows that LR prediction errors are heteroskedastic and non-normal: two of the underlying assumptions are violated. Therefore, LR with ordinary least squares does not seem to be appropriate for the dataset in its current form. While there are methods to tackle the problems associated with violating the underlying assumptions, such as using weighted least squares or transforming the dataset, exploring those potential solutions is left for future projects. The current project is continued by exploring tree-based regression models. Despite the violations in its underlying assumptions, LR was able to reach an R-squared score of about 0.42, and including more than seven features (out of fifteen) in training had virtually no impact on that score. 

Next, tree-based models were explored. The first one was the decision tree. After tunning (with cross-validation) its hyper-parameters, it was found that the only hyper-parameter whose tunning had a significant impact on the prediction scores was maximum depth. In tunning, the search space for a parameter contained at least three values and, through trial and error, it was ensured that the space is wide enough such that the final tunned value is lower/higher than (not equal to) the upper/lower bounds of the search space. The R-squared test score of the decision tree increased smoothly as the number of features increased from two to eight but then it saturated at 0.5 (twenty percent higher than LR). 

Visualizing the tree revealed a business insight drawn from the data. It showed that, regardless of the number of selected features, the first split of the tree is based on whether the place is a private room or an entire home/apartment. For private rooms, the second split is based on the distance to the center. That means that for people who are looking for private rooms, the most important point is how far the room is from the center. For apartments, the second split is based on the number of people that the place can accommodate and not the distance. People who want to stay at an apartment will probably have a car too, so form them, the distance from the center is not as important. 

After training a decision tree, different ensemble methods were explored. The first one was the Random Forest (RF). In tunning its hyper-parameters, it was noticed that the best R_squared score is achieved when the maximum depth of the tress is not limited and a very high number of estimators, n_estimator, is used. In fact, in the tunning trials I performed, the final tunned value of n_estimator was always the upper bound of my search space. Nontheless, n_estimator had to be limited (to less than 200) due to the limitations in available computational resources. Plotting different scores vs. the number of features used in training also showed that, similar to the single tree, R-squared test score of RF increased smoothly as the number of features increased from two to eight but then it saturated at 0.54, eight percent higher than a single tree. However, that accuracy gain was at the cost of more than hundred-fold (~ number of estimators) increase in the computational time.  

The next ensemble methods used were Xtreme Gradient Boosting (XGBoost) and Light Gradient Boosting (LGBoost). These methods use optimized implementations of Gradient Boosting (for details, see the notebook). They were found to be about thirty times faster to train than the RF, but their final R-squared test score were similar. Nonetheless, the fact that they are fast allowed performing a higher-dimensional grid search. That search, however, did not result in any significant improvements in the R-squared test score.

## Conclusions and future work

Different algorithms, including LR, DT, RF, XGBoost, and LGBoost were trained to predict Airbnb rental prices. The highest R-squared test score (around 0.53) was obtained using the ensemble methods (RF, XGBoost, and LGBoost). Among the latter methods, XGBoost and LGBoost were found to be, respectively, thirty times faster to train than RF. The most important feature was found to be the room_type: entire home/apt or private room. The next most important feature for entire home/apts and private rooms were found to be, respectively, the number of people that can be accomodated and the distance from the city center. Althuogh the dataset used for training had 16 features, including features other than 'bathrooms', 'accommodates', 'cleaning_fee', 'room_type', 'distance' did not imporve the score. The score may be further improved by including text features that exist in the dataset in the form of comments or descriptions. This is left for a future project.

## Description of the code structure

The notebook contains an object-oriented code, developed using Scikit-learn, that builds linear and tree-based regression models to predict Airbnb prices. The training dataset contained more than one-hundred features and twenty-thousand observations. The code first drops features that cannot possibly have any predictive value (for example, those containing URLs). It then performs Exploratory Data Analysis. 

The code consists of the following seven classes:

CleanData: contains all the methods responsible for cleaning data, such as: replacing symbols, standardizing letters, and converting data types.
ExploreData: contains methods responsible for exploring data such as printing summary information.
PreprocessData: contains methods responsible for preprocessing data, such as encoding/scaling catagorical/numerical variables 

EDA: This is the class that performs the actual explatory data analysis and it inherits the methods implemented in the above classes. 

SelectFeatures: An "abstraact" class that implements recursive feature elimination. 
BuildModelAfterSelectingFeatures: This class inherits from the SelectFeatures class, performs the actual feature selection, builds the model, performs hyper-parameter tunning, and makes predictions.

EvaluateTheModel: this class implements all the methods responsible for evaluating different models, such as calculating RMSE and r2 scores and methods that check whether the underlying assumptions of linear regress are satisfied. 

Performed feature engineering/selection, Grid Search hyper-parameter tuning, and statistical hypothesis testing.


### Further Readings
XGBoost: 
    Original paper: https://arxiv.org/pdf/1603.02754.pdf
    Documentation: https://xgboost.readthedocs.io/en/latest/
    Nice blog post: https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/

LightGBM:
    Original paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2017/11/lightgbm.pdf
    Documentation: https://lightgbm.readthedocs.io/en/latest/Features.html
    A nice blog post: https://docs.microsoft.com/en-us/archive/blogs/machinelearning/lessons-learned-benchmarking-fast-machine-learning-algorithms

Style: 
    https://legacy.python.org/dev/peps/pep-0008/#other-recommendations