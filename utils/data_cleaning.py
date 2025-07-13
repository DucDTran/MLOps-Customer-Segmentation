
import pandas as pd

def data_cleaning(ecommerce_data):

    # Format transactions data
    ecommerce_df = ecommerce_data.copy()
    ecommerce_df['InvoiceDate'] = pd.to_datetime(ecommerce_df['InvoiceDate'])
    ecommerce_df['CustomerID'] = ecommerce_df['CustomerID'].astype(str)
    ecommerce_df = ecommerce_df[ecommerce_df['UnitPrice'] > 0]
    ecommerce_df['PriceExt'] = ecommerce_df['Quantity'] * ecommerce_df['UnitPrice']

    # Reorder columns
    cols = ['CustomerID', 'InvoiceNo', 'InvoiceDate'] + [col for col in ecommerce_df.columns if col not in ['CustomerID', 'InvoiceNo', 'InvoiceDate']]
    ecommerce_df = ecommerce_df[cols]

    return ecommerce_df