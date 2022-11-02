import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

# books에 category_high를 추가해주는 코드
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


# 작가 preprocessing
def preprocessing_book_author(books:pd.DataFrame) -> pd.DataFrame:
    books['book_author'] = books['book_author'].str.replace('.','', regex=True)
    books['book_author'] = books['book_author'].str.replace('_',' ', regex=True)
    books['book_author'] = books['book_author'].str.lower()
    books['book_author'] = books['book_author'].apply(lambda x:' '.join(sorted(x.split())))
    books['book_author'] = books.book_author.apply(lambda x: re.sub("[\W_]+"," ",x).strip()) # 남은 특수문자 제거 및 strip
    athr_cnt = books.book_author.value_counts().reset_index()
    not1cnt = athr_cnt[athr_cnt.book_author>1]['index'].unique().tolist()
    books.loc[(books.book_author.apply(len)<=2) | (~books.book_author.isin(not1cnt)),'book_author']='others'
    return books


# train에서 평점횟수가 1이하인 책 rating 보정
def edit_once_rated_book(train: pd.DataFrame) -> pd.DataFrame:
    total_avg = train['rating'].mean()
    rating_time = train['isbn'].value_counts()
    rating_time.to_frame()
    rating_time = rating_time.reset_index().rename(columns={'index':'isbn', 'isbn':'cnt'})
    rating_time['cnt']=rating_time['cnt'].astype(str)
    tmp = train.merge(rating_time, how="left", on="isbn")
    tmp.loc[tmp['cnt'] == '1', 'rating'] = tmp['rating']-(tmp['rating']-total_avg)*0.5831
    train = tmp.drop(['cnt'], axis=1)
    return train

# train에서 평점횟수가 1이하인 유저의 rating 보정
def edit_once_rated_user(train: pd.DataFrame) -> pd.DataFrame:
    total_avg = train['rating'].mean()
    rating_time = train['user_id'].value_counts()
    rating_time.to_frame()
    rating_time = rating_time.reset_index().rename(columns={'index':'user_id', 'user_id':'cnt'})
    rating_time['cnt']=rating_time['cnt'].astype(str)
    tmp = train.merge(rating_time, how="left", on="user_id")
    tmp.loc[tmp['cnt'] == '1', 'rating'] = tmp['rating']-(tmp['rating']-total_avg)*0.5831
    train = tmp.drop(['cnt'], axis=1)
    return train

# 미션 1 출판사명 수정함수
def publisher_modify(books):
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])

    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)
    
    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values
    
    for publisher in modify_list:
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass


# location 수정함수 (country nan 행에 대해) : 미션 1
def location_modify_country(users: pd.DataFrame) -> pd.DataFrame:
    users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.
    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values

    location_list = []
    for location in modify_location:
        try:
            right_location = users[(users['location'].str.contains(location, regex=False))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass
    
    for location in location_list:
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]
    return users


# location 수정함수 (state nan 행에 대해)
def location_modify_state(users: pd.DataFrame) -> pd.DataFrame:
    users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.
    modify_location = users[(users['location_state'].isna())&(users['location_city'].notnull())]['location_city'].values

    location_list = []
    for location in modify_location:
        try:
            right_location = users[(users['location'].str.contains(location, regex=False))&(users['location_state'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass
    for location in location_list:
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]
    return users