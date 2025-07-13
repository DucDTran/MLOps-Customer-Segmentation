import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import re

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
colors = ['pink', 'blue', 'red', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'brown']

def text_data_processing(cleaned_data):

    products_raw = cleaned_data[['StockCode', 'Description', 'UnitPrice']].drop_duplicates()

    def get_mode(series):
        return series.mode().iloc[0] if not series.mode().empty else series.iloc[0]

    # Aggregate products
    products_summary = products_raw.groupby('StockCode').agg({
        'Description': lambda x: get_mode(x.dropna()),
        'UnitPrice': 'median'
    }).reset_index()

    products_summary.columns = ['StockCode', 'mode_description', 'median_unit_price']

    # Filter bad stock codes and negative prices
    bad_codes = ['DOT', 'M', 'POST', 'D', 'S', 'AMAZONFEE', 'BANK CHARGES', 'CRUK']
    products_filtered = products_summary[
        (~products_summary['StockCode'].isin(bad_codes)) & 
        (products_summary['median_unit_price'] > 0) &
        (products_summary['mode_description'].notna())
    ].copy()

    products_df = products_filtered

    nltk.download('punkt_tab')
    all_terms = []
    product_terms = {}

    for idx, row in products_df.iterrows():
        description = str(row['mode_description']).lower()
        # Remove special characters and numbers
        description = re.sub(r'[^a-zA-Z\s]', '', description)
        
        # Tokenize
        tokens = word_tokenize(description)
        
        # Remove stop words, colors, and stem
        processed_tokens = []
        for token in tokens:
            if (token not in stop_words and 
                token not in colors and 
                len(token) > 2):
                stemmed = stemmer.stem(token)
                processed_tokens.append(stemmed)
                all_terms.append(stemmed)
        
        product_terms[row['StockCode']] = processed_tokens

    # Get term frequencies
    term_freq = Counter(all_terms)
    top_100_terms = [term for term, count in term_freq.most_common(100)]
    # Create text features matrix
    text_features = []
    stock_codes = []

    for stock_code, terms in product_terms.items():
        features = [terms.count(term) for term in top_100_terms]
        text_features.append(features)
        stock_codes.append(stock_code)

    # Create DataFrame
    text_features_df = pd.DataFrame(text_features, columns=top_100_terms)
    text_features_df['StockCode'] = stock_codes

    # Join with products data
    products_with_features = products_df.merge(text_features_df, on='StockCode')

    return products_df, products_with_features