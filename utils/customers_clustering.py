import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from joblib import dump

def customers_clustering(ecommerce_df, product_clusters_df):

    transactions_with_clusters = ecommerce_df.merge(
        product_clusters_df[['StockCode', 'product_clusters']], 
        on='StockCode', 
        how='left'
    )
    # Customer spend habits
    customer_spend = transactions_with_clusters.groupby(['CustomerID', 'InvoiceNo'])['PriceExt'].sum().reset_index()
    customer_spend_habits = customer_spend.groupby('CustomerID')['PriceExt'].agg([
        'count', 'min', 'mean', 'max', 'sum'
    ]).reset_index()
    customer_spend_habits.columns = ['CustomerID', 'order_count', 'min_order', 'mean_order', 'max_order', 'total_spend']

    # Customer product category preferences
    customer_categories = transactions_with_clusters.groupby(['CustomerID', 'product_clusters'])['PriceExt'].sum().reset_index()
    customer_categories_pivot = customer_categories.pivot(
        index='CustomerID', 
        columns='product_clusters', 
        values='PriceExt'
    ).fillna(0)

    # Calculate proportions
    customer_categories_pivot = customer_categories_pivot.div(
        customer_categories_pivot.sum(axis=1), axis=0
    ).fillna(0)
    # Rename columns
    customer_categories_pivot.columns = [f'cat_{int(col)}' for col in customer_categories_pivot.columns]
    customer_categories_pivot = customer_categories_pivot.reset_index()

    # Customer recency
    max_date = transactions_with_clusters['InvoiceDate'].max()
    customer_recency = transactions_with_clusters.groupby('CustomerID')['InvoiceDate'].agg(['min', 'max']).reset_index()
    customer_recency['first_purchase_days'] = (max_date - customer_recency['min']).dt.days
    customer_recency['last_purchase_days'] = (max_date - customer_recency['max']).dt.days
    customer_recency = customer_recency[['CustomerID', 'first_purchase_days', 'last_purchase_days']]

    # Join all customer features
    customer_features = customer_spend_habits.merge(customer_categories_pivot, on='CustomerID')
    customer_features = customer_features.merge(customer_recency, on='CustomerID')

    # Select features for clustering
    feature_cols = [col for col in customer_features.columns if col not in ['CustomerID', 'min_order', 'max_order']]
    print(feature_cols)
    X_customers = customer_features[feature_cols].copy()

    # Log transform skewed features
    log_features = ['order_count', 'mean_order', 'total_spend', 'first_purchase_days', 'last_purchase_days']
    for col in log_features:
        if col in X_customers.columns:
            X_customers[col] = np.log1p(X_customers[col])

    # Scale features
    scaler = StandardScaler()
    X_customers_scaled = scaler.fit_transform(X_customers)

    X_customers = X_customers_scaled
    X_customers = np.nan_to_num(X_customers, nan=0)
    customer_ids = customer_features['CustomerID'].values

    # K-Means clustering
    kmeans = KMeans(n_clusters=5, random_state=84)
    customer_clusters = kmeans.fit_predict(X_customers)

    # t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=84)
    X_embedded = tsne.fit_transform(X_customers)

    # Create customer clusters dataframe
    customer_clusters_df = pd.DataFrame({
        'CustomerID': customer_ids,
        'customer_clusters': customer_clusters,
        'tsne_1': X_embedded[:, 0],
        'tsne_2': X_embedded[:, 1]
    })

    # Add customer features
    customer_clusters_df = customer_clusters_df.merge(
        customer_features, on='CustomerID'
    )

    dump(scaler, 'artifacts/objects/customers_scaler.joblib')
    dump(kmeans, 'artifacts/objects/customers_kmeans.joblib')

    return customer_clusters_df