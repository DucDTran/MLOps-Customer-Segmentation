def customers_radar(customer_clusters_df):
    # Calculate cluster morphology
    numeric_cols = ['order_count', 'mean_order', 'total_spend', 'first_purchase_days', 'last_purchase_days']
    category_cols = [col for col in customer_clusters_df.columns if col.startswith('cat_')]

    cluster_morphology = customer_clusters_df.groupby('customer_clusters')[numeric_cols + category_cols].mean()

    return cluster_morphology