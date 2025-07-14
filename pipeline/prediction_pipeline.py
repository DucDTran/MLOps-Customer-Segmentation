
import pandas as pd
import numpy as np
from joblib import load

CUSTOMER_SCALER = load('artifacts/objects/customers_scaler.joblib')

def generate_features_for_customer(customer_df: pd.DataFrame, 
                                   all_transactions_df: pd.DataFrame, 
                                   product_clusters_df: pd.DataFrame):

    print(customer_df.info())
    print(customer_df)
    order_count = customer_df['InvoiceNo'].nunique()
    total_spend = customer_df['PriceExt'].sum()
    mean_order = customer_df.groupby('InvoiceNo')['PriceExt'].sum().mean() if order_count > 0 else 0
    
    all_transactions_df['InvoiceDate'] = pd.to_datetime(all_transactions_df['InvoiceDate'], errors='coerce')
    customer_df['InvoiceDate'] = pd.to_datetime(customer_df['InvoiceDate'], errors='coerce')

    max_date = all_transactions_df['InvoiceDate'].max()
    first_purchase_days = (max_date - customer_df['InvoiceDate'].min()).days
    last_purchase_days = (max_date - customer_df['InvoiceDate'].max()).days
    
    customer_with_clusters = customer_df.merge(product_clusters_df, on='StockCode', how='left')
    customer_with_clusters['product_clusters'].fillna(-1, inplace=True)
    category_spend = customer_with_clusters.groupby('product_clusters')['PriceExt'].sum()
    category_props = (category_spend / category_spend.sum()).rename(lambda x: f"cat_{int(x)}")
    
    all_feature_names = CUSTOMER_SCALER.get_feature_names_out()
    feature_vector = pd.Series(0.0, index=all_feature_names)
    
    new_data = {
        'order_count': order_count,
        'total_spend': total_spend,
        'mean_order': mean_order,
        'first_purchase_days': first_purchase_days,
        'last_purchase_days': last_purchase_days,
    }
    new_data.update(category_props)
    
    feature_vector.update(pd.Series(new_data))
    
    for col in ['order_count', 'mean_order', 'total_spend', 'first_purchase_days', 'last_purchase_days']:
        if col in feature_vector.index:
            feature_vector[col] = np.log1p(feature_vector[col])
            
    scaled_features = CUSTOMER_SCALER.transform(feature_vector.values.reshape(1, -1))
    return scaled_features