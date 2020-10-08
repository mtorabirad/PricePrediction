# Predicting Airbnb prices using different Machine Learning algorithms


## Summary

In this project, I have shown how to build different Machine Learning models to predict Airbnb prices. The project is motivated by the following business question: "What dictates Airbnb rental price? The number of beds? The number of guests allowed? Distance from the center? Review score? Cancellation policy?" 

After cleaning/preprocessing it, Exploratory Data Analysis (EDA) was performed to identify outliers and possible inter-correlation between the features. Linear and tree-based regression models are trained. It was found that Linear Regression (LR) prediction errors are heteroskedastic and non-normal, indicating that LR is not suitable to fit the training data in its current format. Despite the violations in its underlying assumptions, LR was able to reach an R-squared score of around 0.4, and including more than four features (out of fifteen) in training had virtually no impact on that score. 

## Instructions 

If you would like to run the notebook, first download the csv file (containing Airbnb listings.csv) and the set the variable "CSV_local_address" in the second cell of the notebook to the local addrress of the downloaded file.

## Requirements
Python 3.6
Scikit-learn
Pandas
Numpy
seaborn
scipy
statsmodels

## Detailed Description
The training dataset contained more than one-hundred features and twenty-thousand observations. The code first drops features that cannot possibly have any predictive value (for example, those containing URLs). It then performs the Exploratory Data Analysis (EDA). In EDA, outliers are detected and handled using established statistical procedures. The presence of inter-correlations between the features are examined, and domain knowledge is used to select a single feature among the ones with high inter-correlation and also to engineer a new feature. Categorical features that contain a natural order ('room_type' and 'cancelation policy') are encoded using ordinal encoding, and all the other categorical features are encoded using pandas get_dummies. Because the data set was low-cardinality, there was no need to use more advanced encoding methods. Numerical features are scaled using the normalization method. Other scaling methods were also explored, but their influence on the results was found to be negligible.

After EDA, a hold-out set is first prepared and set aside. Then, different Ml models are explored. The first model is the linear regression (LR). The predictions of LR are analyzed in detail to check whether they satisfy the four underlying assumptions of LR: errors should have 1) zero mean and 2) constant variance (homoskedastic) and be 3) uncorrelated and 4) normally distributed. The analysis shows that LR prediction errors are heteroskedastic and non-normal: two of the underlying assumptions are violated. Therefore, LR with ordinary least squares does not seem to be appropriate for the dataset in its current form. While there are methods to tackle the problems associated with violating the underlying assumptions, such as using weighted least squares or transforming the dataset, exploring those potential solutions is left for future projects. The current project is continued by exploring tree-based regression models. 

The first tree-based model used is the decision tree. After tunning (with cross-validation) its hyper-parameters, it was found that tunning only the maximum depth hyper-parameter has a significant impact on the prediction scores. Visualizing the tree revealed a business insight that can be drawn from the data. It showed that, regardless of the number of selected features, the first split of the tree is based on whether the place is a private room or an entire apartment. For private rooms, the second split is based on the distance to the center. That means that for people who are looking for private rooms, the most important point is how far the room is from the center. For apartments, the second split is based on the number of people that the place can accommodate and not the distance. People who want to stay at an apartment will probably have a car too, so form them, the distance from the center is not as important. 

Depending on the number of selected features, the r2 score of the decision tree was observed to be between 0.35 and 0.4, whereas that score for linear regression was 0.4. 

After training a decision tree, different ensemble methods were explored. The first one is the Random Forest (RF). Because tunning hyper-parameters of a decision tree had already shown that only the maximum depth hyper-parameter is worth tunning, for the random forest, only the maximum depth, number of features, and number of estimators hyper-parameters were tunned. Despite these tunning, the r2 score of the RF was found to be only a few percent higher than the single decision tree. Plotting the score vs. the number of features used also showed that including more than 6 features (out of 15) does not result in any gain in the predictive power of the model. 

The next ensemble methods used were Xtreme Gradient Boosting (XGBoost) and Light Gradient Boosting (LGBoost). These methods use optimized implementations of Gradient Boosting (for details, see the notebook). It was found that their maximum scores, which was reached when eight features were selected, are almost the same as the random forest. However, the computational times of these optimized methods were almost half of the random forest. 

All the algorithms explored in this project (linear regression, decision tree, random forest, and gradient boosting) were able to reach R-squared scores of between 0.45 and 0.5, and including more than six features (out of fifteen) did not improve the prediction scores. This suggests that a higher score can only be achieved if text features that exist in the dataset are also included in the training. This is left for a future project.



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

Recommendations for style: https://legacy.python.org/dev/peps/pep-0008/#other-recommendations