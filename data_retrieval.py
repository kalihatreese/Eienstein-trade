import pandas as pd
import requests

def retrieve_historical_data(symbol):
    try:
        data = pd.read_csv('data.csv')
        data.set_index('Date', inplace=True)
        return data[symbol]
    except Exception as e:
        print(f"Error: {e}")
        return None

def retrieve_sentiment_data_from_liquid_nero(symbol, start_date, end_date):
    api_key = 'YOUR_API_KEY'
    endpoint = f'https://api.liquidnero.com/sentiment/{symbol}?start={start_date}&end={end_date}'
    headers = {'Authorization': f'Bearer {api_key}'}

    response = requests.get(endpoint, headers=headers)
    if response.status_code == 200:
        sentiment_data = response.json()
        return sentiment_data
    else:
        print(f"Error: {response.status_code}")
        return None
