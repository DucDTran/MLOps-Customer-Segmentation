import numpy as np
from sklearn.preprocessing import StandardScaler

def feature_engineering(dataframe):
    feature_cols = ['order_count', 'mean_order', 'total_spend', 'first_purchase_days', 'last_purchase_days'] + [col for col in dataframe.columns if col.startswith('cat_')]

    X = dataframe[feature_cols].copy()
    y = dataframe['customer_clusters'].copy()

    # Log transform and scale
    X['mean_order'] = np.log1p(X['mean_order'])
    X['total_spend'] = np.log1p(X['total_spend'])
    X['first_purchase_days'] = np.log1p(X['first_purchase_days'])
    X['last_purchase_days'] = np.log1p(X['last_purchase_days'])
    X['order_count'] = np.log1p(X['order_count'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_prediction = X_scaled
    X_prediction = np.nan_to_num(X_prediction, nan=0)
    y_prediction = y.values

    return X_prediction, y_prediction