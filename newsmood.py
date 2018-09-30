#Import Dependencies

import tweepy
import pandas as pd
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from datetime import datetime

# Twitter API Keys
from config import (consumer_key, 
                    consumer_secret, 
                    access_token, 
                    access_token_secret)

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())


# Import Vader and set up Sentiment analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# Initializing target users as twitter handles creating empty lists
target_list = ["nytimes", "FoxNews", "BBC","CBS", "CNN"]

tweets_ago = []
compound = []
positive = []
negative = [] 
neutral = []
created = []
source = []


# Api queries and appending empty lists


for target in target_list:
    public_tweets = api.user_timeline(target, count=100,result_type = "recent")
    
    i=0
    for tweets in public_tweets:
         compound.append(analyzer.polarity_scores(tweets["text"])["compound"])
         positive.append(analyzer.polarity_scores(tweets['text'])["pos"])
         negative.append(analyzer.polarity_scores(tweets['text'])["neg"])
         neutral.append(analyzer.polarity_scores(tweets["text"])["neu"])
         created.append(datetime.strptime(tweets["created_at"], "%a %b %d %H:%M:%S %z %Y"))
         source.append(tweets["user"]["screen_name"])
         tweets_ago.append(i)
         
         i =i+1


# Creating and saving dataframes to Csv

news_dict = {"Source" : source,
             "Compound_Score": compound,
             "Positive_score": positive,
             "Negative_Score" : negative,
             "Neutral_Score": neutral,
             "Created_at": created,
             "Tweets Ago": tweets_ago
}

news_data= pd.DataFrame(news_dict)
news_data.to_csv("Newsdata_senti",index=False)
news_data.head()

# Set index to source
news_data.set_index('Source').head()


# Scatter Plot Variables

tweets_nytimes = news_data.loc[news_data["Source"] == 'nytimes',"Tweets Ago"]
comp_nytimes = news_data.loc[news_data["Source"] == "nytimes", "Compound_Score"]

tweets_cnn = news_data.loc[news_data["Source"] == 'CNN',"Tweets Ago"]
comp_cnn = news_data.loc[news_data["Source"] == "CNN", "Compound_Score"]

tweets_cbs = news_data.loc[news_data["Source"] == 'CBS',"Tweets Ago"]
comp_cbs = news_data.loc[news_data["Source"] == "CBS", "Compound_Score"]

tweets_bbc = news_data.loc[news_data["Source"] == 'BBC',"Tweets Ago"]
comp_bbc = news_data.loc[news_data["Source"] == "BBC", "Compound_Score"]

tweets_foxnews = news_data.loc[news_data["Source"] == 'FoxNews',"Tweets Ago"]
comp_foxnews = news_data.loc[news_data["Source"] == "FoxNews", "Compound_Score"]

import datetime
plt.figure(figsize=(13,7))
plt.scatter(tweets_nytimes,comp_nytimes,marker="o",color="yellow",edgecolors="black",alpha= 1.00)
plt.scatter(tweets_cnn,comp_cnn,marker="o",color="red",edgecolors="black",alpha= 1.00)
plt.scatter(tweets_bbc,comp_bbc,marker="o",color="lightblue",edgecolors="black",alpha= 1.00)
plt.scatter(tweets_cbs,comp_cbs,marker="o",color="darkgreen",edgecolors="black",alpha= 1.00)
plt.scatter(tweets_foxnews,comp_foxnews,marker="o",color="blue",edgecolors="black",alpha= 1.00)
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Popularity")

currentdt = datetime.datetime.now()
date = currentdt.strftime("%m/%d/%y")
plt.title(f"Sentiment Analysis of Media tweets as of {date}")
plt.savefig("Sentiment_analysis")


# Bar Chart Variables
overall_t_nytimes = np.mean(tweets_nytimes)
overall_c_nytimes = np.mean(comp_nytimes)
overall_t_cnn = np.mean(tweets_cnn)
overall_c_cnn = np.mean(comp_cnn)
overall_t_cbs = np.mean(tweets_cbs)
overall_c_cbs = np.mean(comp_cbs)
overall_t_bbc = np.mean(tweets_bbc)
overall_c_bbc = np.mean(comp_bbc)
overall_t_foxnews = np.mean(tweets_foxnews)
overall_c_foxnews = np.mean(comp_foxnews)      



#Plotting Bar Charts


x = [0,1,2,3,4]

plt.figure(figsize=(12,7))
plt.bar(0, overall_c_nytimes, facecolor='yellow', alpha=0.6, align="center",width=1.0)
plt.bar(1, overall_c_cnn, facecolor='red', alpha=1.0, align="center",width=1.0)
plt.bar(2, overall_c_cbs, facecolor='darkgreen', alpha=1.0, align="center",width=1.0)
plt.bar(3, overall_c_bbc, facecolor='lightblue', alpha=1.0, align="center",width=1.0)
plt.bar(4, overall_c_foxnews, facecolor='blue', alpha=1.0, align="center",width=1.0)
plt.xlabel("News Sources")
plt.ylabel("Tweet Popularity")
currentdt = datetime.datetime.now()
date = currentdt.strftime("%m/%d/%y")
plt.title(f"Overall Media Sentiment Based On Twitter  ({date})")

tick = [value for value in x]
plt.xticks(tick, ["NY Times","CNN","CBS", "BBC", "FoxNews"], fontsize=15, color='black')
plt.savefig("Overall_sentiment")


