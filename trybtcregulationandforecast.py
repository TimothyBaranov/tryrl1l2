import ccxt
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

bybit = ccxt.bybit({
    'enableRateLimit': True,
})


def fetch_ohlcv(symbol, timeframe, since, limit):
    ohlcv = bybit.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    data = [[x[0], x[1], x[2], x[3], x[4], x[5]] for x in ohlcv]
    return pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])


symbol = 'BTC/USDT'
timeframe = '1d'
since = bybit.parse8601('2022-01-01T00:00:00Z')
limit = 100

df = fetch_ohlcv(symbol, timeframe, since, limit)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)


X = df[['open', 'high', 'low', 'volume']].values
y = df['close'].values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
ridge_errors = []
lasso_errors = []

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)
    ridge_mse = mean_squared_error(y_test, ridge_pred)
    ridge_errors.append(ridge_mse)

    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train, y_train)
    lasso_pred = lasso_model.predict(X_test)
    lasso_mse = mean_squared_error(y_test, lasso_pred)
    lasso_errors.append(lasso_mse)


plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_errors, marker='o', label='Ridge')
plt.plot(alphas, lasso_errors, marker='o', label='Lasso')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('Effect of Alpha on Ridge and Lasso Regression Performance')
plt.legend()
plt.show()
