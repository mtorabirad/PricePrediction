import seaborn as sns, numpy as np
import matplotlib.pyplot as plt


#sns.set_theme(); 
np.random.seed(0)
x = np.random.randn(100)
ax = sns.distplot(x, norm_hist=False, kde=False)
plt.show()
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=7)
from lightgbm import LGBMRegressor

LGBMRegressor()
import xgboost as xgb

model = xgb.XGBRegressor() 
