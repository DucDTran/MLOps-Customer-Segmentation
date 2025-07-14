import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import pandas as pd
from joblib import dump

def products_clustering(products_df, products_with_features):

    # Select numeric features
    feature_cols = [col for col in products_with_features.columns 
                    if col not in ['StockCode', 'mode_description']]

    X_products = products_with_features[feature_cols].copy()

    # Log transform and scale price
    X_products['median_unit_price'] = np.log1p(X_products['median_unit_price'])

    # Scale features
    scaler = StandardScaler()
    X_products_scaled = scaler.fit_transform(X_products)

    X_products = X_products_scaled
    product_stock_codes = products_with_features['StockCode'].values

    # K-Means clustering
    kmeans = KMeans(n_clusters=5, random_state=84)
    product_clusters = kmeans.fit_predict(X_products)

    # t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=84)
    X_embedded = tsne.fit_transform(X_products)

    # Create product clusters dataframe
    product_clusters_df = pd.DataFrame({
        'StockCode': product_stock_codes,
        'product_clusters': product_clusters,
        'tsne_1': X_embedded[:, 0],
        'tsne_2': X_embedded[:, 1]
    })

    # Add product info
    product_clusters_df = product_clusters_df.merge(
        products_df[['StockCode', 'median_unit_price', 'mode_description']], 
        on='StockCode'
    )

    dump(scaler, 'artifacts/objects/products_scaler.joblib')
    dump(kmeans, 'artifacts/objects/products_kmeans.joblib')

    product_clusters_df.to_csv('artifacts/data/product_clusters.csv', index=False)

    return product_clusters_df