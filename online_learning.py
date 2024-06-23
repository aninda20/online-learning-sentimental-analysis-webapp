import streamlit as st
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    html_temp = """
    <div style="background-color:DodgerBlue;"><p style="color:white;font-size:30px;padding:9px">Sentimental analysis on Online learning using Social media during Covid-19</p></div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.subheader("Select an online learning topic in wordcloud which you'd like to get the sentiment analysis on:")

    # Load the data from the uploaded CSV file
    data_path = 'data/tweets_sentiments.csv'
    df = pd.read_csv(data_path)

    # Rename columns to match expected structure
    df = df.rename(columns={
        'Content': 'Tweet',
        'Username': 'User',
        'Created at': 'Date',
        'Favorites': 'Likes',
        'Retweet-Count': 'RT',
        'Location': 'User_location',
        'Label': 'Sentiment'
    })

    def clean_tweet(tweet):
        return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|([RT])', ' ', tweet.lower()).split())

    def analyze_sentiment(tweet):
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0:
            return 'Positive'
        elif analysis.sentiment.polarity == 0:
            return 'Neutral'
        else:
            return 'Negative'

    def prepCloud(Topic_text, Topic):
        Topic = str(Topic).lower()
        Topic = ' '.join(re.sub('([^0-9A-Za-z \t])', ' ', Topic).split())
        Topic = re.split("\s+", str(Topic))
        stopwords = set(STOPWORDS)
        stopwords.update(Topic)
        text_new = " ".join([txt for txt in Topic_text.split() if txt not in stopwords])
        return text_new

    image = Image.open('images/Logo1.JPG')
    st.image(image, caption='Sentimental analysis on Online learning during Covid-19', use_column_width=True)
    
    Topic = str(st.text_input("Enter the topic from the word cloud you are interested in (e.g., online course), and press Enter once done."))
    if Topic in ["distance learning", "online teaching", "online education", "online course", "online semester", "distance course", "distance education", "online class", "e-learning", "e learning", "#distancelearning", "#onlineschool", "#onlineteaching", "#virtuallearning", "#onlineducation", "#distanceeducation", "#OnlineClasses", "#DigitalLearning", "#elearning", "#onlinelearning"]:
        if len(Topic) > 0:
            with st.spinner("Please wait, Tweets are being extracted"):
                filtered_df = df[df['Tweet'].str.contains(Topic, case=False, na=False)]
                st.success('Tweets have been Extracted !!!!')

            filtered_df['clean_tweet'] = filtered_df['Tweet'].apply(lambda x: clean_tweet(x))
            filtered_df["Sentiment"] = filtered_df['Tweet'].apply(lambda x: analyze_sentiment(x))

            st.write("Total Tweets Extracted for Topic '{}' are : {}".format(Topic, len(filtered_df.Tweet)))
            st.write("Total Positive Tweets are : {}".format(len(filtered_df[filtered_df["Sentiment"] == "Positive"])))
            st.write("Total Negative Tweets are : {}".format(len(filtered_df[filtered_df["Sentiment"] == "Negative"])))
            st.write("Total Neutral Tweets are : {}".format(len(filtered_df[filtered_df["Sentiment"] == "Neutral"])))

            if st.button("See the Extracted Data"):
                st.success("Below is the Extracted Data :")
                st.write(filtered_df.head(250))

            if st.button("Get Count Plot for Different Sentiments"):
                st.success("Generating A Count Plot")
                st.subheader(" Count Plot for Different Sentiments")
                st.write(sns.countplot(filtered_df["Sentiment"]))
                st.pyplot()

            if st.button("Get Pie Chart for Different Sentiments"):
                st.success("Generating A Pie Chart")
                a = len(filtered_df[filtered_df["Sentiment"] == "Positive"])
                b = len(filtered_df[filtered_df["Sentiment"] == "Negative"])
                c = len(filtered_df[filtered_df["Sentiment"] == "Neutral"])
                d = np.array([a, b, c])
                explode = (0.1, 0.0, 0.1)
                st.write(plt.pie(d, shadow=True, explode=explode, labels=["Positive", "Negative", "Neutral"], autopct='%1.2f%%'))
                st.pyplot()

            if st.button("Get WordCloud for all things said about {}".format(Topic)):
                st.success("Generating A WordCloud for all things said about {}".format(Topic))
                text = " ".join(review for review in filtered_df.clean_tweet)
                stopwords = set(STOPWORDS)
                text_newALL = prepCloud(text, Topic)
                wordcloud = WordCloud(stopwords=stopwords, max_words=800, max_font_size=70).generate(text_newALL)
                st.write(plt.imshow(wordcloud, interpolation='bilinear'))
                st.pyplot()

            if st.button("Get WordCloud for all Positive Tweets about {}".format(Topic)):
                st.success("Generating A WordCloud for all Positive Tweets about {}".format(Topic))
                text_positive = " ".join(review for review in filtered_df[filtered_df["Sentiment"] == "Positive"].clean_tweet)
                stopwords = set(STOPWORDS)
                text_new_positive = prepCloud(text_positive, Topic)
                wordcloud = WordCloud(stopwords=stopwords, max_words=800, max_font_size=70).generate(text_new_positive)
                st.write(plt.imshow(wordcloud, interpolation='bilinear'))
                st.pyplot()

            if st.button("Get WordCloud for all Negative Tweets about {}".format(Topic)):
                st.success("Generating A WordCloud for all Negative Tweets about {}".format(Topic))
                text_negative = " ".join(review for review in filtered_df[filtered_df["Sentiment"] == "Negative"].clean_tweet)
                stopwords = set(STOPWORDS)
                text_new_negative = prepCloud(text_negative, Topic)
                wordcloud = WordCloud(stopwords=stopwords, max_words=800, max_font_size=70).generate(text_new_negative)
                st.write(plt.imshow(wordcloud, interpolation='bilinear'))
                st.pyplot()

        if st.button("Exit"):
            st.balloons()

if __name__ == '__main__':
    main()
