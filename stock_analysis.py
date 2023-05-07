import pandas as pd
import yfinance as yf
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import datetime
import re
from sklearn.metrics import mean_absolute_error
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def get_tweets(ticker, start_date, end_date):
    if ticker == "AAPL":
        name = "Apple"
    if ticker == "MSFT":
        name = "Microsoft"
    if ticker == "TSLA":
        name = "Tesla"
    if ticker == "AMZN":
        name = "Amazon"
    if start_date != "2015-01-01 00:00:00" or end_date != "2019-12-31 00:00:00":
        tweets = pd.read_csv(f"{name}_Tweets.csv")
        if isinstance(start_date, datetime.datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        start_date_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, datetime.datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        end_date_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        tweets_data = [{'date': tweet["date"], 'text': tweet["text"]} for _, tweet in tweets.iterrows()
                    if start_date_dt <= datetime.datetime.fromtimestamp(tweet["date"]) <= end_date_dt]
    else:
        tweets = tweets = pd.read_csv(f"{name}_Tweets.csv")
        tweets_data = [{'date': tweet["date"], 'text': tweet["text"]} for _, tweet in tweets.iterrows()]
    return tweets_data

def preprocess_tweet(tweet):
    tweet = str(tweet)
    tweet = tweet.lower()
    tweet = tweet.strip()    
    tweet = re.sub(r"http\S+", "", tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tokens = word_tokenize(tweet)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


def analyze_tweets(tweets):
    print("starting analyze_tweets")
    sia = SentimentIntensityAnalyzer()
    print("analyze_tweets: preprocessed tweets gets text from tweets")
    preprocessed_tweets = [preprocess_tweet(tweet['text']) for tweet in tweets]
    print("analyze_tweets: sentiment_scores")
    sentiment_scores = [sia.polarity_scores(tweet)["compound"] for tweet in preprocessed_tweets]
    print(sentiment_scores)
    return sentiment_scores


def collect_stock_data(ticker, start, end):
    print("starting collect_stock_data")
    print("opening csv")
    
    company_values_df = pd.read_csv("companyvalues.csv")
    
    company_values_df.dropna(subset=['change_in_close'], inplace=True)

    # Convert the 'day_date' column to datetime objects
    company_values_df['day_date'] = pd.to_datetime(company_values_df['day_date'])

    print("finding close values for tickers inputted in the date range")
    stock_prices = company_values_df[(company_values_df["ticker_symbol"] == ticker) &
                                     (start <= company_values_df["day_date"]) &
                                     (company_values_df["day_date"] <= end)]
    print("stock prices found")
    return stock_prices


def analyze_stocks(companies, start_date, end_date, model_type):
    if model_type == "all":
        model_types = ["linear_regression", "elastic_net", "svr", "random_forest"]
    else:
        model_types = [model_type]
        
    results = []
    print("starting analyze_stocks")

    for company_name in companies:
        print(f"analyzing {company_name}")
        print("getting company data, opening company.csv")
        company_data = pd.read_csv("company.csv")
        print("finding the ticker")
        ticker = company_data[company_data['company_name'] == company_name]['ticker_symbol'].values[0]
        print(ticker)
        print("finding tweets")
        ## tweets is a list of dictionaries with 'date' and 'text'
        tweets = get_tweets(ticker, start_date, end_date)
        
        print(f"found tweets")
        print("finding sentiment scores")
            
        sentiment_scores = analyze_tweets(tweets)
        print(f"found scores: {sentiment_scores}")
        print("finding stock prices")
        stock_data = collect_stock_data(ticker, start_date, end_date)
        stock_prices = stock_data['change_in_close']
        print(f"found prices: {stock_prices}")

        for tweet in range(0, len(tweets)):
            tweets[tweet]['sentiment'] = sentiment_scores[tweet]
            row = stock_data.loc[stock_data['day_date'] == datetime.datetime.fromtimestamp(tweets[tweet]['date']).strftime('%Y-%m-%d')]
            tweets[tweet]['stock_price'] = row.loc[row.index[0], 'change_in_close']
            
        data = pd.DataFrame(tweets, columns=['sentiment', 'stock_price'])

        print(data)

        # Train and evaluate a Linear Regression model
        X = data["sentiment"].values.reshape(-1, 1)
        y = data["stock_price"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        for model in model_types:
            # Train and evaluate the selected model
            if model == 'linear_regression':
                model = LinearRegression()
                model.fit(X_train, y_train)
            elif model == 'elastic_net':
                model = ElasticNet()
                model.fit(X_train, y_train)
            elif model == 'svr':
                model = SVR()
                model.fit(X_train, y_train)
            elif model == 'random_forest':
                model = RandomForestRegressor()
                model.fit(X_train, y_train)

            # Evaluate the model
            print("calculating predictions")
            y_pred = model.predict(X_test)
            print("calculating scores")
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            if mape != "inf":
                mape = round(mape, 2)
            if mape == "inf":
                mape = "error"
                
            da = sum((y_test - y_pred) >= 0) / len(y_test)
            results.append({"company": company_name, "ticker": ticker, "rmse": round(rmse, 2), "r2": round(r2, 2), "mae": round(mae, 2), "mape": mape, "da": round(da, 2), "model": model})
    print(results)
    return results