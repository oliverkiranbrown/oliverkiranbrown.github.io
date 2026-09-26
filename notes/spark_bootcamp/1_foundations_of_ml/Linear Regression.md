Date: 16-09-2025
# Linear Regression
## Why it matters
Oldie but goldie! If the assumptions are satisfied, then LR should be your baseline/benchmark before moving on to more complex models. Simple, interpretable, intuitive. 
## Assumptions & Preconditions
- **Linearity**: the coefficients are real numbers. You can try polynomial regression if you want to relax this. 
- **Non-multicollinearity** Features must be independent of one-another. 
## How it works (intuition)
Given $n$ data points in $\mathbb{R}^p$, find the $\beta\in\mathbb{R}^{p+1}$ which minimises the loss on this training set. We want to understand how $y$ changes if I take one step forward in $x_i$ holding all other directions fixed. In one dimension...
$$
y=\beta_0+\beta_1 x_1+\beta_2 x_2+...++\beta_p x_p
$$
$\beta$ is found using the MLE (max the log-likelihood by differentiating and finding the values of $\beta$ where $l$ is stationary). See [the docs](https://scikit-learn.org/stable/modules/linear_model.html) for the full formalisation. 
## Minimal Recipe
You'll likely need to do some wrangling first (see tips and tricks). Then implement
```python
import sklearn.model_selection import test_train_split
import sklearn.linear_model import LinearRegression

# prepare features
df = pd.read_csv('path/to/data.csv')
# >>> wrangling / scaling >>>
y = df[['target_var']] # what you want to predict
X = df.drop('target_var', axis=1) # features could be everything but the target

# split data,instantiate and 'train' model (just MLE)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=...)
lr = LinearRegression()
lr.fit(X_train, y_train)

# inspect the coefficents - what features matter most?
coeff_df = pd.DataFrame({'feature': X.columns, 'Coef': lr.coef_})
coeff_df.loc[len(coeff_df)] = ['intercept', lr.intercept_]
coeff_df.sort_values(by='Coef', key=abs, ascending=False)

# does model perform well on unseen data?
y_pred = lr.predict(X_test)
```
## Pitfalls 
- Don't use when relationships between variables aren't linear
- Very sensitive to outliers (make sure to normalise data to minimise the effect)
- If features are co-linear, you can get unstable weight estimates. 
- If data is discrete (e.g. counting etc), reach for other basic distributions, (e.g. [Poisson](https://omarfsosa.github.io/poisson_regression_in_python))
## Metrics & Checks
```python
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_pred, y_test)
rmse = np.sqrt(mse)
r_2 = r2_score(y_pred, y_test)
```
## Tools & Workflows

- Quick and dirty one-hot-encoding of a categorical feature: 
```python
df_one_hot = pd.get_dummies(df, columns=['col1'] # then select these columns as features
```
- Dates to integers: 
```python
df['date'] = pd.to_datetime(df['date'])
df.loc[:, 'day'] = (df['date'] - df['date'].min()).dt.days # feature - days since start
```

- Normalise the data to $\text{N}(0,1)$ via $Z=\frac{X-\mu}{\sigma}$ (where appropriate) for smaller, more comparable values and reduced sensitivity to outliers. Many ways to do this for non-gaussian data. 

```python
from sklearn.preprocessing import StandardScaler

# create the scaler
scaler - StandardScaler() # could do a MinMax etc
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# convert back to a df
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
```

- **Regularisation**: Occam's razor says you want the simplest, most interpretable model possible. To penalise the size of model weights, you can add a *Ridge* or *Lasso* penalty term to the loss function.
	- [**Ridge**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html): $\text{Loss}=\sum_{i=1}^n(y_i-\hat{y})^2+\alpha\sum_{j=1}^p\beta_j^2$  
		- **Behaviour**: gentle reduction in model weights (because $\partial L/\partial \beta$ is linear in $\beta$, weight updates slow the closer you get to zero). 
		- **Use-case**: Useful when all the features are important. 
	- [**Lasso**](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html): $\text{Loss}=\sum_{i=1}^n(y_i-\hat{y})^2+\alpha\sum_{j=1}^p|\beta_j|$
		- **Behaviour**: $\partial L/\partial \beta$ is constant so you have much more aggressive weight updates, pushing some of the least important $\beta_j$ to zero.
		- **Use-case**: useful for auto-feature selection
		- **Warning**: can be unstable. If features are highly correlated, lasso would arbitrarily choose one of the co-linear features. Better to use elastic (combo of lasso and ridge). 
	- **Finding $\alpha$**: Take your dataset and test/train split. With the train set, further split into train and validation. Test different $\alpha$ (via grid search/optimising etc) and find the best performing on the validation set. Then retrain the model using this optimal $\alpha$ on the full training dataset and evaluate on test. 




