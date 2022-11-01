import time, os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from .src.models._models import rmse 

DATA_PATH = './data/'

users = pd.read_csv(DATA_PATH + 'users.csv')
books = pd.read_csv(DATA_PATH + 'books.csv')
train = pd.read_csv(DATA_PATH + 'train_ratings.csv')
test = pd.read_csv(DATA_PATH + 'test_ratings.csv')
sub = pd.read_csv(DATA_PATH + 'sample_submission.csv')

### 그냥 train 으로 하면 메모리 초과...? 클러스터링을 이용해보자!

def IBCF_predict_rating(train, test):
    
    rating_list=[]
    
    books = pd.read_csv(DATA_PATH + 'books.csv')
    books2 = books[['isbn','book_title']]
    train_df = pd.merge(train, books2, how = 'left', on = 'isbn')
    
    user_item_matrix = train_df.pivot_table(index=['book_title'], columns=['user_id'], values='rating')
    user_item_matrix = user_item_matrix.fillna(0)
    item_similarity_df = pd.DataFrame(cosine_similarity(user_item_matrix), index=user_item_matrix.index, columns=user_item_matrix.index)

    for test_id, test_isbn in zip(test['user_id'], test['isbn']):
        test_book_title = train_df['isbn' == test_isbn]['book_title']
        try:
            pred_rating = (item_similarity_df[test_book_title].sort_index().values * user_item_matrix[test_id].sort_index().values).sum() / item_similarity_df[test_book_title].values.sum()
        except:
            pred_rating=0
        rating_list.append(pred_rating)
        
    return rating_list


predicts = IBCF_predict_rating(train, test)

submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')
submission['rating'] = predicts

now = time.localtime()
now_date = time.strftime('%Y%m%d', now)
now_hour = time.strftime('%X', now)
save_time = now_date + '_' + now_hour.replace(':', '')
os.makedirs('submit', exist_ok=True)
submit_file_path = 'submit/{}_IBCF.csv'.format(save_time)
submission.to_csv(submit_file_path, index=False)
print(f"Submit File Saved: {submit_file_path}")
