Date: 24-09-2025
#  Time Series Analysis with SARIMAX
## Why it matters
[SARIMAX](https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html) is a sensible baseline model for modelling time series data. It should be a 'first go' before reaching for more complex tools. 
## Assumptions & Preconditions
- Stationary data — the statistical properties of the time series must remain constant over time (for a ARMA model)
- Seasonality removed / accounted for (ARIMA model)
- Residuals should be white noise so that all meaningful patterns are captured by the model. If there's still structure, you might need to reach for a more expressive model family.
## How it works (intuition)
You assume you can breakdown a time series into the trend (long-term progression), seasonality (recurring patterns) and noise (residual). Note, this could be [additive or multiplicative](https://kourentzes.com/forecasting/2014/11/09/additive-and-multiplicative-seasonality/) depending on the situation. 
$$
y_t=T_t+S_t+R_t
$$
First, you want to remove the trend to make the series *stationary* (constant mean, variance, covariance over time) by differencing. You take one term from another:
- **First-order**: $\Delta y_t=y_t-y_{t-1}$
- **Consecutive**: $\Delta^2 y_t=y_t-2y_{t-1}-y_{t-2}$
- **Seasonal**: $\Delta_p y_t=y_t-y_{t-p}$ (e.g. $p=7$ for weekly data)

With a stationary series, you can now model with autoregressive (AR) and moving-average (MA) terms:
- **AR**: the autoregressive terms are simply a linear combination of the previous time steps (lags). 
	- **Formally**: An $\text{AR}(p)$ model takes the form $y_t=\sum_{i=1}^p\phi_i y_{t-i}+\epsilon_t$ where the $\epsilon_t$ is random noise with mean 0 and variance $\sigma^2$.
	- **Determining lags**: You can use Autocorrelation and Partial Autocorrelation (ACF/PACF) plots to determine the number of lags to look back. 
		- ACF looks at all terms; PACF controls for the terms before so you count only it's impact. 
- **MA**: you try to predict the current value based on past errors. 
	- **Formally**: An $\text{MA}(q)$ model takes the form $y_t=\mu+\sum_{j=1}^q\theta_j\epsilon_{t-j}+\epsilon_t$ where $\mu$ is the mean of the series and the $\epsilon_{t-j}$ are the error terms. 
	- Useless on it's own, but when paired with $\text{AR}(p)$, you have...
- **ARMA**: you can capture both the lagged relationship and the dependence on past errors by combining
	- **Formally**: $\text{ARMA}(p,q)=\sum_{i=1}^p\phi_i y_{t-i}+\sum_{j=1}^q\theta_j\epsilon_{t-j}+\epsilon_t$
- **ARIMA**: In an *Autoregressive Integrated Moving Average* model, $\text{ARIMA}(p,d,q)$, you take the $d$th difference $\Delta^d y_t$ and model the resulting time series via $\text{ARMA}(p,q)$. 
	- See details [here](https://www.stats.ox.ac.uk/~reinert/time/notesht10short.pdf). 
- **SARIMA**: add in seasonality differencing to the *AR*, *MA* and *I* terms to form $\text{ARIMA}((p,d,q)\times(P,D,Q)_m)$. 
	- $y_t=\text{ARIMA}_{\text{short}}(p,d,q) + \text{ARIMA}_{\text{seasonal}}(P,D,Q)_{\text{period}=m}+\epsilon_t$
- **SARIMAX**: finally, add in eXogenous factors to the modelling
## Minimal Recipe

```python
import pmdarima as pm
from statsmodels.tsa.statespace import SARIMAX
import pmdarima as pm

df = pd.read_csv('path/to/data/')
df_diff = df.diff(4) # difference the series to stationarity as req

# Find optimal parameters by applying ADF/KPSS automatically
arima_params = pm.auto_arima(df_diff,
	start_p=2, start_q=2,
	max_p=5, max_q=5, # use ACF/PACF plots to determine sensible range
	seasonal=True,
	m=24, # e.g. seasonal component happens every 24 hours
	start_P=1, start_Q=1,
	max_P=2, max_Q=2
	)
# extract the best values from this internal grid search
best_order = arima_params.get_params()["order"]
best_seasonal_order = arima_params.get_params()["seasonal_order"]

# instantiate and train the model
sarima_model = SARIMAX(
	df,
	order=best_order, # ARIMA(p, d, q)
	seasonal_order=best_seasonal_order # SARIMA(P, D, Q, m)
)
sarima_result = sarima_model.fit()
# predict 
forcast = arima_model.get_forecast(steps=24) # predict two days in advance
```
## Metrics & Checks
You must be careful to avoid adding too many autoregressive terms and overfitting your model to the data. Define the **Akaike Information Criterion** (AIC) and **Bayesian Information Criterion** (BIC) as
$$
\text{AIC}=2k-2\log(L)
$$
$$
\text{BIC}=k\log(n)-2\log(L)
$$
where 
- $k$: the number of parameters in the model
- $n$: number of observations
- $L$: likelihood of the model (how well it fits to data)

**Goal:** As low a score as possible. The first terms punish overfitting (BIC more than AIC), the second terms measure performance. 
## Tools & Workflows

- **Decomposition** of series into a trend, seasonal term and residuals

```python
from statsmodels.tsa.seasonal import seasonal_decomposition
df = pd.read_csv('/path/to/data/')
df = df.asfreq('h', method=ffill) # converts to an hourly time series
decomposition = seasonal_decomposition(df, model='additive', period=24)
decomposition.plot(); # breakdown into components for exploratory thinkikng
```

- **Checking for Stationarity**: There's two statistical tests you can run. Each looks from a different vantage point. 
	- **Augmented Dickey Fuller (ADF)** — hypothesis test where you check for the presence of a [unit root]((https://faculty.washington.edu/ezivot/econ584/notes/unitroot.pdf) which would imply the series is non-stationary.
	- **Kwiatkowski-Phillips-Schmidt-Shin (KPSS)** — more simply checks whether the series is stationary around a deterministic trend. 

```python
from statsmodels.tsa.stattools import adfuller, kpss
adf_result = addfuller(time_series_data)
kpss_result = kpss(time_series_data, regression='c') # assume a constant trend, can use 'ct' if broadly linear
```

See the [docs](https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html) for implementation and test interpretation details. The series is only ready for modelling when both tests indicate stationarity. Keep differencing till you get there via 

- **Autocorrelation and Partial Autocorrelation (ACF/PACF) plots**.
	- ACF asks, how correlated is today's value with everything that came $k$ periods ago?
	- PACF asks, how correlated are today's values with the period exactly $k$ periods ago, controlling for everything that came before. 

```python
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
plot_acf(df, lags=24*14) # 14 days ago to look for weekly trends
plot_pacf(df, lags=24*2) # look two days ago
```

- Finding the **optimal lags** in a more basic setting. BIC tends to prefer simpler models that avoid overfitting

```python
from statsmodels.tsa.ar_model import ar_select_order
mod = ar_select_order(df, maxlag=10, ic='bic') # can specifiy the infomation criterion
optlag = max(mod.ar_lags)
```

