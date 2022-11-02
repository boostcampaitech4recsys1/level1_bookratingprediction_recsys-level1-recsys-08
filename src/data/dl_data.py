import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from .utils import make_category_high, preprocessing_book_author, \
                    edit_once_rated_book, edit_once_rated_user, publisher_modify, \
                    location_modify_country, location_modify_state
from .context_data import age_map


# train에 feature merge해주는 함수
def train_merge_feature(users, books, train, user_feat_list, book_feat_list):
    # 피쳐 전처리
    if 'book_author' in set(book_feat_list):
        books = preprocessing_book_author(books)
    if 'publisher' in set(book_feat_list):
        books = publisher_modify(books)
    # if 'location' in set(book_feat_list):
    #     if len(set(users.columns).intersection(location_set))==3: # 기존 users에 city, state, country가 존재한다면,
    #         pass
    #     else:
    #         users['location_city'] = users['location'].apply(lambda x: x.split(',')[0].strip())
    #         users['location_state'] = users['location'].apply(lambda x: x.split(',')[1].strip())
    #         users['location_country'] = users['location'].apply(lambda x: x.split(',')[2].strip())
    #         # 🍁🍁🍁 location 전처리, 주의❗️ 아래의 두 함수를 호출하면 데이터 로드가 약 1분 30초가 소요됨.
    #         users = location_modify_country(users)
    #         users = location_modify_state(users)
    #         users['location_city'] = users['location_city'].map(loc_city2idx)
    #         users['location_state'] = useres['location_state'].map(loc_state2idx)
    #         users['location_country'] = users['location_country'].map(loc_country2idx)
    
    # 피쳐 인덱싱
    users['age'] = users['age'].fillna(int(users['age'].mean()))
    users['age'] = users['age'].apply(age_map)
    category2idx = {v:k for k,v in enumerate(books['category'].unique())}
    categoryhigh2idx = {v:k for k,v in enumerate(books['category_high'].unique())}
    publisher2idx = {v:k for k,v in enumerate(books['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(books['language'].unique())}
    author2idx = {v:k for k,v in enumerate(books['book_author'].unique())}

    books['category'] = books['category'].map(category2idx)
    books['category_high'] = books['category_high'].map(categoryhigh2idx)
    books['publisher'] = books['publisher'].map(publisher2idx)
    books['language'] = books['language'].map(language2idx)
    books['book_author'] = books['book_author'].map(author2idx)

    if user_feat_list:
        train = train.merge(users[['user_id']+user_feat_list], how='left', on='user_id')
    if book_feat_list:
        train = train.merge(books[['isbn']+book_feat_list], how='left', on='isbn')
    return train
        

def dl_data_load(args):
    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    # 🍁🍁🍁 books에 category_high 추가
    books = make_category_high(books)

    # train에 feature merge해주는 함수
    train = train_merge_feature(users, books, train, user_feat_list=['age'], book_feat_list=[])

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)

    field_dims = np.array([len(user2idx), len(isbn2idx), 7], dtype=np.uint32)

    data = {
            'train':train,
            'test':test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }

    return data

def dl_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def dl_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
