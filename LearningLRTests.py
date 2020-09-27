from sklearn.linear_model import LinearRegression
from sklearn import tree
tree.plot_tree(model)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_depth=2, random_state=0)
from subprocess import call
from statsmodels.stats.diagnostic import normal_ad
import seaborn as sns 
sns.distplot
sns.distplot(self.model_residuals, norm_hist=True)