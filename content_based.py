import re
import time, os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def make_category_high(books:pd.DataFrame) -> pd.DataFrame:
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    books['category'] = books['category'].str.lower()
    categories = ['garden','crafts','physics','adventure','music','fiction','nonfiction','science','science fiction','social','homicide',
    'sociology','disease','religion','christian','philosophy','psycholog','mathemat','agricult','environmental',
    'business','poetry','drama','literary','travel','motion picture','children','cook','literature','electronic',
    'humor','animal','bird','photograph','computer','house','ecology','family','architect','camp','criminal','language','india']

    books['category_high'] = books['category'].copy()
    for category in categories:
        books.loc[books[books['category'].str.contains(category,na=False)].index,'category_high'] = category
        
    category_high_df = pd.DataFrame(books['category_high'].value_counts()).reset_index()
    category_high_df.columns = ['category','count']
    others_list = category_high_df[category_high_df['count']<5]['category'].values
    books.loc[books[books['category_high'].isin(others_list)].index, 'category_high']='others'
    return books

def content_data_set():
    
    DATA_PATH = './data/'
    users = pd.read_csv(DATA_PATH + 'users.csv')
    books = pd.read_csv(DATA_PATH + 'books.csv')
    train = pd.read_csv(DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')
    
    # books = make_category_high(books)
    # books_df = pd.merge(train, books[['isbn', 'book_title', 'language', 'category_high']], how = 'left', on = 'isbn')
    books_df = pd.merge(train, books[['isbn', 'book_title']], how = 'left', on = 'isbn')
    test_df = pd.merge(test, books[['isbn', 'book_title']], how = 'left', on = 'isbn')
    
    return books_df, test_df


books_df, test_df = content_data_set()
counter_vector = CountVectorizer(ngram_range=(1, 3))
c_vector_title = counter_vector.fit_transform(books_df['book_title'])
similarity_title = cosine_similarity(c_vector_title, c_vector_title).argsort()[:,::-1]


def recommend_book_list(df, book_title, top=5):
  target_isbn = df[df['book_title'] == book_title].index.values

  sim_index = similarity_title[target_isbn, :top].reshape(-1)
#   sim_index = sim_index[sim_index!=target_isbn]

  result = df.iloc[sim_index].sort_values('rating', ascending=False)

  return result


rating_list = []
for test_isbn, test_title in zip(test_df['isbn'], test_df['book_title']):

  if test_title.isna() :
      rating_list.append(books_df['rating'].mean())
      continue

  try:
      result = recommend_book_list(books_df, test_title)
      rating = result['rating'].mean()
  except:
      pred_rating = books_df['rating'].mean()
      
  rating_list.append(pred_rating)

submission = pd.read_csv('./data/' + 'sample_submission.csv')
submission['rating'] = rating_list

now = time.localtime()
now_date = time.strftime('%Y%m%d', now)
now_hour = time.strftime('%X', now)
save_time = now_date + '_' + now_hour.replace(':', '')
os.makedirs('submit', exist_ok=True)
submit_file_path = 'submit/{}_IBCF.csv'.format(save_time)
submission.to_csv(submit_file_path, index=False)
print(f"Submit File Saved: {submit_file_path}")