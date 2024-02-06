import pandas as pd
import requests
from google.colab import drive
drive.mount('/content/drive')
from bs4 import BeautifulSoup
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import gspread
from google.oauth2 import service_account
from google.colab import auth
!gcloud config set project new-assignment-413121
!gcloud services enable drive.googleapis.com
# Authenticate with Google Colab
auth.authenticate_user()
credentials_file_path = '/content/drive/MyDrive/new-assignment.json'

# Scopes required for accessing Google Sheets
scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

# Authenticate with Google using the downloaded credentials file
credentials = service_account.Credentials.from_service_account_file(credentials_file_path, scopes=scopes)

# Authorize access to Google Sheets
gc = gspread.authorize(credentials)

# Web Scraping
cnbc_url = "https://www.cnbc.com/search/?query=green%20hydrogen&qsearchterm=green%20hydrogen"
cnbc_page = requests.get(cnbc_url)
cnbc_soup = BeautifulSoup(cnbc_page.content, 'html.parser')
cnbc_headlines = [headline.text.strip() for headline in cnbc_soup.find_all("div", class_="SearchResultCard-headline")]

google_news_url = "https://news.google.com/rss/search?q=green%20hydrogen&hl=en-IN&gl=IN&ceid=IN:en"
google_news_page = requests.get(google_news_url)
google_news_soup = BeautifulSoup(google_news_page.content, 'xml')
google_news_headlines = [item.title.text for item in google_news_soup.find_all('item')]

# Create a Pandas DataFrame with Headlines and Dates
df = pd.DataFrame({'Headline': cnbc_headlines + google_news_headlines,
                   'Date': [datetime.today().strftime('%Y-%m-%d')] * len(cnbc_headlines) +
                           [item.pubDate.text for item in google_news_soup.find_all('item')]})

# Sentiment Analysis
sentiment_analysis = pipeline("sentiment-analysis")
df['Sentiment_Score'] = df['Headline'].apply(lambda x: sentiment_analysis(x)[0]['score'])

# Named Entity Recognition (NER)
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_model = pipeline("ner", model=model, tokenizer=tokenizer)
df['Organizations'] = df['Headline'].apply(lambda x: [ent['word'] for ent in ner_model(x) if ent['entity'] in ['B-ORG', 'I-ORG',  'B-LOC', 'I-LOC']])
# Post-process organization names
df['Organizations'] = df['Organizations'].apply(lambda org_list: ' '.join([org.replace('##', '') for org in org_list if org.isalpha()]))

# Convert the list of organizations to a string before updating Google Sheet
df['Organizations_String'] = df['Organizations']

# Export to CSV
df.to_csv('green_hydrogen_news.csv', index=False)

# Google Sheets API
# Create a New Google Sheet for Data Storage
sheet_name = 'Green_Hydrogen_News_Analysis'
worksheet = gc.create(sheet_name).get_worksheet(0)

# Write Processed Data to Google Sheet
data_to_update = [df.columns.values.tolist()] + df.drop(columns=['Organizations']).values.tolist()
worksheet.update(data_to_update)

# Drop the temporary column used for string representation of organizations
df = df.drop(columns=['Organizations_String'])


# Week-wise Sentiment Trend Graph
df['Date'] = pd.to_datetime(df['Date'])
weekly_trend = df.resample('W-Mon', on='Date').mean()

plt.plot(weekly_trend.index, weekly_trend['Sentiment_Score'])
plt.xlabel('Week')
plt.ylabel('Average Sentiment Score')
plt.title('Week-wise Sentiment Trend')
plt.show()
# Word Cloud Map
all_organizations = [org for sublist in df['Organizations'].dropna() for org in sublist]
if all_organizations:
    wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(' '.join(all_organizations))
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title('Word Cloud Map of Organizations in News Headlines')
    plt.show()
else:
    print("No organizations found for Word Cloud.")
