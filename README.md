# sentiment_analysis
#libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df_tweets  = pd.read_csv('clean_tweet_state.csv')
df_geomap = pd.read_csv('geoMap.csv')
df_ggtrend = pd.read_csv('gg_trend_2020.csv')
df_event = pd.read_csv('HCQ_event.csv')

df_ggtrend['date'] = pd.to_datetime(df_ggtrend['Day'])
df_ggtrend.drop(['Day'], axis=1, inplace=True)
df_tweets['date'] = pd.to_datetime(df_tweets['created_at'])
df_tweets.drop(['created_at'], axis=1, inplace=True)
df_event['date'] = pd.to_datetime(df_event['date'])

df_tweets_counts = df_tweets.groupby('date')['full_text'].count().div(2000)
df_tweets_favorites = df_tweets.groupby('date')['favorite_count'].sum().div(100000)
df_tweets_combined = pd.concat([df_tweets_counts, df_tweets_favorites], axis=1)

df_trend_scores = df_ggtrend.groupby('date')['score'].sum()

fig, axs = plt.subplots(nrows=2, figsize=(12,10))

axs[0].plot(df_tweets_combined.index, df_tweets_combined['full_text'], color='blue')
axs[0].plot(df_tweets_combined.index, df_tweets_combined['favorite_count'], color='orange')
axs[0].set_title('Tweets and Favorites per Day')
axs[0].set_ylabel('Number of tweets (in hundred-thousands)')

ax2 = axs[0].twinx()
ax2.set_ylabel('Number of favorites (in two-thousands)')
ax2.tick_params(axis='y', labelcolor='orange')

axs[1].plot(df_trend_scores.index, df_trend_scores.values, color='green')
axs[1].set_title('Google Trend Score')
axs[1].set_ylabel('Google Trend Score')

for i, row in df_event.iterrows():
    date = row['date']
    event = row['event']
    if date in df_trend_scores.index:
        score = df_trend_scores[date]
        axs[1].annotate(event, xy=(date, score), xytext=(date, score+3), ha='center', fontsize=5, va='center', rotation=6)

plt.subplots_adjust(hspace=0.3)
plt.xticks(rotation=45)

plt.show()
