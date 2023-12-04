
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# %matplotlib inline

"""## Train dataset used for our analysis"""

train = pd.read_csv('https://drive.google.com/file/d/1ZGTfgxxw-O9KBa8bNEVZhOKv3I2P_YgD/view?usp=sharing')

"""#### We make a copy of training data so that even if we have to make any changes in this dataset we would not lose the original dataset."""

train_original=train.copy()

"""#### Here we see that there are a total of 31692 tweets in the training dataset"""

train.shape

train_original

"""## Test dataset used for our analysis"""

test = pd.read_csv('https://drive.google.com/file/d/1joaXdPu62sgWt2zgkB8ksdNdEaiUOykO/view?usp=sharing')

"""#### We make a copy of test data so that even if we have to make any changes in this dataset we would not lose the original dataset."""

test_original=test.copy()

"""#### Here we see that there are a total of 17197 tweets in the test dataset"""

test.shape

test_original

"""### We combine Train and Test datasets for pre-processing stage"""

combine = train.append(test,ignore_index=True,sort=True)

combine.head()

combine.tail()

"""# Data Pre-Processing
"""

def remove_pattern(text,pattern):

    r = re.findall(pattern,text)

    for i in r:
        text = re.sub(i,"",text)

    return text

combine['Tidy_Tweets'] = np.vectorize(remove_pattern)(combine['tweet'], "@[\w]*")

combine.head()

"""## Removing Punctuations, Numbers, and Special Characters
"""

combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")

combine.head(10)

"""## Removing Short Words
"""

combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

combine.head(10)

"""## Tokenization
"""

tokenized_tweet = combine['Tidy_Tweets'].apply(lambda x: x.split())
tokenized_tweet.head()

"""## Stemming
"""

from nltk import PorterStemmer

ps = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

tokenized_tweet.head()

"""#### Now let’s stitch these tokens back together."""

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combine['Tidy_Tweets'] = tokenized_tweet
combine.head()

"""# Visualization from Tweets

"""

from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import urllib
import requests

""" #### Store all the words from the dataset which are non-racist/sexist"""

all_words_positive = ' '.join(text for text in combine['Tidy_Tweets'][combine['label']==0])

"""#### We can see most of the words are positive or neutral. With happy, smile, and love being the most frequent ones. Hence, most of the frequent words are compatible with the sentiment which is non racist/sexists tweets."""

Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

image_colors = ImageColorGenerator(Mask)

wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(all_words_positive)

plt.figure(figsize=(10,20))

plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")

plt.axis('off')
plt.show()

"""#### Store all the words from the dataset which are racist/sexist"""

all_words_negative = ' '.join(text for text in combine['Tidy_Tweets'][combine['label']==1])

"""#### As we can clearly see, most of the words have negative connotations. So, it seems we have a pretty good text data to work on."""

Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

image_colors = ImageColorGenerator(Mask)

wc = WordCloud(background_color='black', height=1500, width=4000,mask=Mask).generate(all_words_negative)

plt.figure(figsize=(10,20))

plt.imshow(wc.recolor(color_func=image_colors),interpolation="gaussian")

plt.axis('off')
plt.show()

"""# Understanding the impact of Hashtags on tweets sentiment
### Function to extract hashtags from tweets
"""

def Hashtags_Extract(x):
    hashtags=[]

    for i in x:
        ht = re.findall(r'#(\w+)',i)
        hashtags.append(ht)

    return hashtags

"""#### A nested list of all the hashtags from the positive reviews from the dataset"""

ht_positive = Hashtags_Extract(combine['Tidy_Tweets'][combine['label']==0])

"""#### Here we unnest the list"""

ht_positive_unnest = sum(ht_positive,[])

"""#### A nested list of all the hashtags from the negative reviews from the dataset"""

ht_negative = Hashtags_Extract(combine['Tidy_Tweets'][combine['label']==1])

"""#### Here we unnest the list"""

ht_negative_unnest = sum(ht_negative,[])

"""## Plotting BarPlots
### For Positive Tweets in the dataset

#### Counting the frequency of the words having Positive Sentiment
"""

word_freq_positive = nltk.FreqDist(ht_positive_unnest)

word_freq_positive

"""#### Creating a dataframe for the most frequently used words in hashtags"""

df_positive = pd.DataFrame({'Hashtags':list(word_freq_positive.keys()),'Count':list(word_freq_positive.values())})

df_positive.head(10)

"""#### Plotting the barplot for the 10 most frequent words used for hashtags"""

df_positive_plot = df_positive.nlargest(20,columns='Count')

sns.barplot(data=df_positive_plot,y='Hashtags',x='Count')
sns.despine()

"""### For Negative Tweets in the dataset

#### Counting the frequency of the words having Negative Sentiment
"""

word_freq_negative = nltk.FreqDist(ht_negative_unnest)

word_freq_negative

"""#### Creating a dataframe for the most frequently used words in hashtags"""

df_negative = pd.DataFrame({'Hashtags':list(word_freq_negative.keys()),'Count':list(word_freq_negative.values())})

df_negative.head(10)

"""#### Plotting the barplot for the 10 most frequent words used for hashtags"""

df_negative_plot = df_negative.nlargest(20,columns='Count')

sns.barplot(data=df_negative_plot,y='Hashtags',x='Count')
sns.despine()

"""# Extracting Features from cleaned Tweets

### Bag-of-Words Features

"""

from sklearn.feature_extraction.text import CountVectorizer

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')

bow = bow_vectorizer.fit_transform(combine['Tidy_Tweets'])

df_bow = pd.DataFrame(bow.todense())

df_bow

"""### TF-IDF Features
"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')

tfidf_matrix=tfidf.fit_transform(combine['Tidy_Tweets'])

df_tfidf = pd.DataFrame(tfidf_matrix.todense())

df_tfidf

"""# Applying Machine Learning Models

"""

train_bow = bow[:31962]

train_bow.todense()

"""### Using features from TF-IDF for training set"""

train_tfidf_matrix = tfidf_matrix[:31962]

train_tfidf_matrix.todense()

"""### Splitting the data into training and validation set"""

from sklearn.model_selection import train_test_split

"""#### Bag-of-Words Features"""

x_train_bow,x_valid_bow,y_train_bow,y_valid_bow = train_test_split(train_bow,train['label'],test_size=0.3,random_state=2)

"""#### Using TF-IDF features"""

x_train_tfidf,x_valid_tfidf,y_train_tfidf,y_valid_tfidf = train_test_split(train_tfidf_matrix,train['label'],test_size=0.3,random_state=17)

"""
## Logistic Regression"""

from sklearn.linear_model import LogisticRegression

Log_Reg = LogisticRegression(random_state=0,solver='lbfgs')

"""### Using Bag-of-Words Features"""

# Fitting the Logistic Regression Model

Log_Reg.fit(x_train_bow,y_train_bow)

# The first part of the list is predicting probabilities for label:0
# and the second part of the list is predicting probabilities for label:1
prediction_bow = Log_Reg.predict_proba(x_valid_bow)

prediction_bow

"""#### Calculating the F1 score"""

from sklearn.metrics import f1_score

# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
prediction_int = prediction_bow[:,1]>=0.3

prediction_int = prediction_int.astype(np.int)
prediction_int

# calculating f1 score
log_bow = f1_score(y_valid_bow, prediction_int)

log_bow

"""### Using TF-IDF Features"""

Log_Reg.fit(x_train_tfidf,y_train_tfidf)

prediction_tfidf = Log_Reg.predict_proba(x_valid_tfidf)

prediction_tfidf

"""#### Calculating the F1 score"""

prediction_int = prediction_tfidf[:,1]>=0.3

prediction_int = prediction_int.astype(np.int)
prediction_int

# calculating f1 score
log_tfidf = f1_score(y_valid_tfidf, prediction_int)

log_tfidf

"""## XGBoost"""

from xgboost import XGBClassifier

"""### Using Bag-of-Words Features"""

model_bow = XGBClassifier(random_state=22,learning_rate=0.9)

model_bow.fit(x_train_bow, y_train_bow)

# The first part of the list is predicting probabilities for label:0
# and the second part of the list is predicting probabilities for label:1
xgb=model_bow.predict_proba(x_valid_bow)

xgb

"""#### Calculating the F1 score"""

# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
xgb=xgb[:,1]>=0.3

# converting the results to integer type
xgb_int=xgb.astype(np.int)

# calculating f1 score
xgb_bow=f1_score(y_valid_bow,xgb_int)

xgb_bow

"""### Using TF-IDF Features"""

model_tfidf=XGBClassifier(random_state=29,learning_rate=0.7)

model_tfidf.fit(x_train_tfidf, y_train_tfidf)

# The first part of the list is predicting probabilities for label:0
# and the second part of the list is predicting probabilities for label:1
xgb_tfidf=model_tfidf.predict_proba(x_valid_tfidf)

xgb_tfidf

"""#### Calculating the F1 score"""

# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
xgb_tfidf=xgb_tfidf[:,1]>=0.3

# converting the results to integer type
xgb_int_tfidf=xgb_tfidf.astype(np.int)

# calculating f1 score
score=f1_score(y_valid_tfidf,xgb_int_tfidf)

score

"""## Decision Tree"""

from sklearn.tree import DecisionTreeClassifier

dct = DecisionTreeClassifier(criterion='entropy', random_state=1)

"""### Using Bag-of-Words Features"""

dct.fit(x_train_bow,y_train_bow)

dct_bow = dct.predict_proba(x_valid_bow)

dct_bow

# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
dct_bow=dct_bow[:,1]>=0.3

# converting the results to integer type
dct_int_bow=dct_bow.astype(np.int)

# calculating f1 score
dct_score_bow=f1_score(y_valid_bow,dct_int_bow)

dct_score_bow

"""### Using TF-IDF Features"""

dct.fit(x_train_tfidf,y_train_tfidf)

dct_tfidf = dct.predict_proba(x_valid_tfidf)

dct_tfidf

"""#### Calculating F1 Score"""

# if prediction is greater than or equal to 0.3 than 1 else 0
# Where 0 is for positive sentiment tweets and 1 for negative sentiment tweets
dct_tfidf=dct_tfidf[:,1]>=0.3

# converting the results to integer type
dct_int_tfidf=dct_tfidf.astype(np.int)

# calculating f1 score
dct_score_tfidf=f1_score(y_valid_tfidf,dct_int_tfidf)

dct_score_tfidf

"""# Model Comparison"""

Algo=['LogisticRegression(Bag-of-Words)','XGBoost(Bag-of-Words)','DecisionTree(Bag-of-Words)','LogisticRegression(TF-IDF)','XGBoost(TF-IDF)','DecisionTree(TF-IDF)']

score = [log_bow,xgb_bow,dct_score_bow,log_tfidf,score,dct_score_tfidf]

compare=pd.DataFrame({'Model':Algo,'F1_Score':score},index=[i for i in range(1,7)])

compare.T

plt.figure(figsize=(18,5))

sns.pointplot(x='Model',y='F1_Score',data=compare)

plt.title('Model Vs Score')
plt.xlabel('MODEL')
plt.ylabel('SCORE')

plt.show()

"""## Using the best possible model to predict for the test data

#### From the above comaprison graph we can see that Logistic Regression trained using TF-IDF features gives us the best performance
"""

test_tfidf = tfidf_matrix[31962:]

test_pred = Log_Reg.predict_proba(test_tfidf)

test_pred_int = test_pred[:,1] >= 0.3

test_pred_int = test_pred_int.astype(np.int)

test['label'] = test_pred_int

submission = test[['id','label']]

submission.to_csv('result.csv', index=False)

"""### Test dataset after prediction"""

res = pd.read_csv('result.csv')

res


sns.countplot(train_original['label'])
sns.despine()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import re

# Function to preprocess the new post
def preprocess_post(post):
    # Remove special characters, URLs, and user mentions
    post = re.sub(r'@[A-Za-z0-9]+', '', post)
    post = re.sub('https?://[A-Za-z0-9./]+', '', post)
    post = re.sub("[^a-zA-Z]", " ", post)
    post = ' '.join([w for w in post.split() if len(w) > 3])

    # Apply stemming
    post_tokens = post.split()
    post_tokens = [ps.stem(token) for token in post_tokens]
    post = ' '.join(post_tokens)

    return post

# Function to predict sentiment of a new post
def predict_sentiment(post, model, vectorizer):
    # Preprocess the post
    preprocessed_post = preprocess_post(post)

    # Vectorize the preprocessed post using the trained TF-IDF vectorizer
    post_vectorized = vectorizer.transform([preprocessed_post])

    # Make predictions using the trained Logistic Regression model
    prediction = model.predict_proba(post_vectorized)

    # Output the sentiment prediction
    if prediction[0, 1] >= 0.3:  # Adjust the threshold as needed
        return "Negative"
    else:
        return "Positive"

# Example usage
new_post = "Idli Sambar is Good"
sentiment = predict_sentiment(new_post, Log_Reg, tfidf)  # Use Log_Reg instead of log_Reg
print(f"The sentiment of the post is: {sentiment}")