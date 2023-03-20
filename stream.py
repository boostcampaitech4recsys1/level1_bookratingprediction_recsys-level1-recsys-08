import pickle
import re
import numpy as np
import pandas as pd
from typing import List, Set, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import streamlit as st
import torch
from PIL import Image
import pickle
from src import NeuralCollaborativeFiltering
from src.data import dl_data_load, dl_data_split, dl_data_loader

st.set_page_config(page_title="Books Recommender", layout="wide")

## 함수부
def content_based_filtering_euclidean(book_title_list: list, 
                       wordmatrix: np.array, # TF-IDF로 계산한 vectorization array
                       title: str, # title과 유사도가 높은 애들 출력
                       topn: Optional[int]=None) -> pd.DataFrame:
    topn=11 if topn is None else topn+1
    sim_matrix = pd.DataFrame(euclidean_distances(wordmatrix), index=book_title_list, columns=book_title_list)
    target_similarity_df = sim_matrix[title].reset_index().copy()
    target_similarity_df.columns=['title', 'euclidean_similarity']
    return target_similarity_df.sort_values('euclidean_similarity', ascending=True).reset_index(drop=True)[1:topn]


## UserID, ISBN Prediction
with open('data.pickle','rb') as f:
    data = pickle.load(f)
model_path = 'models/20221110_121250_NCF_MODEL.pt'
with open('config.pickle','rb') as f:
    args = pickle.load(f)
model = NeuralCollaborativeFiltering(args, data, inference_model=model_path)

max_user = max(data['idx2user'].keys())
max_isbn = max(data['idx2isbn'].keys())
with st.expander("If You Know UserID and BookID - Predict Score"):
    user_id = st.number_input('Pick User ID', 0, max_user)
    book_id = st.number_input('Pick Book ID', 0, max_isbn)
    predict_score = model.predict_once(torch.tensor([user_id, book_id], dtype=torch.int32))
    st.write(f"Predict Score is: {predict_score.item():.2f}")

def capture_return(_):
    st.session_state["clicked"] = True

def add_book(book_id):
    if st.session_state["clicked"]:
        st.session_state["selected_book_count"] += 1
        st.session_state["added_book_ids"].append(book_id)
        st.session_state["clicked"] = False

def set_value(key):
    st.session_state[key] = st.session_state["key_" + key]

def set_status(status):
    st.session_state["status"] = status

def retry():
    st.session_state["selected_book_count"] = 0
    st.session_state["added_book_ids"] = []
    st.session_state["status"] = False

STATE_KEYS_VALS = [
    ("selected_book_count", 0),  # main part
    ("added_book_ids", []),  # main part
    ("status", False),
    ("clicked", False),
    ("input_len", 5),  # sidebar
    ("top_k", 10),  # sidebar
    ("years", (1990, 2010)),  # sidebar
]
for k, v in STATE_KEYS_VALS:
    if k not in st.session_state:
        st.session_state[k] = v


## Contents Based Collaborative Filtering
top_books = pd.read_csv("data/top_books.csv")
with open('wordmatrix.pickle','rb') as f:
    wordmatrix = pickle.load(f)

## Side Bar
val = st.sidebar.number_input(
    "How many books do you want to choose?",
    format="%i",
    min_value=5,
    max_value=20,
    value=int(st.session_state["input_len"]),
    disabled=st.session_state["status"],
    on_change=set_value,
    args=("input_len",),
    key="key_input_len",
)

st.sidebar.button(
    "START",
    on_click=set_status,
    args=(True,),
    disabled=st.session_state["status"],
)

st.title("BookTube")

if st.session_state["status"]:
    unique_key = 0
    # top_books2 = top_books[~top_books.isbn.isin(st.session_state["added_book_ids"])].reset_index(drop=True)
    selection_container = st.empty()
    if st.session_state["selected_book_count"] < st.session_state["input_len"]:
        with selection_container.container():
            st.subheader(
                "Please select the favorite Book: {}/{}".format(
                    st.session_state["selected_book_count"],
                    st.session_state["input_len"],
                )
            )
            for row_index in range(2):
                for col_index, col in enumerate(st.columns(10)):
                    unique_key += 1
                    with col:
                        st.image(top_books.loc[unique_key,'img_url'])
                        capture_return(
                                        st.checkbox(
                                            top_books.loc[unique_key,'book_title'],
                                            key=f"{top_books.loc[unique_key,'isbn']}",
                                            on_change=add_book,
                                            args=(top_books.loc[unique_key,"isbn"],),
                                        )
                                    )
    else:
        ## Empty the above view
        selection_container.empty()

        ## Top 10
        top10_isbn = top_books.sort_values('Popularity',ascending=False).isbn[:10].tolist()
        top10_books = top_books[top_books.isbn.isin(top10_isbn)].reset_index(drop=True)
        
        ## Recommend
        isbn_list = st.session_state.added_book_ids
        print(">>>",isbn_list)
        title_list = top_books[top_books.isbn.isin(isbn_list)].book_title.tolist()
        df_recommend=pd.DataFrame(columns=['title','euclidean_similarity'])
        for title in title_list:
            df = content_based_filtering_euclidean(top_books.book_title.tolist(), wordmatrix, title=title,topn=5)
            df_recommend=pd.concat([df_recommend,df])
        df_recommend.drop_duplicates(inplace=True)
        df_recommend = df_recommend[~df_recommend.title.isin(title_list)]
        df_recommend.euclidean_similarity = df_recommend.euclidean_similarity - (df_recommend.euclidean_similarity.max()-1)
        df_recommend.rename(columns={'title':'book_title','euclidean_similarity':'rec_score'},inplace=True)
        df_recommend.sort_values('rec_score',ascending=False, inplace=True)
        df_recommend.reset_index(drop=True,inplace=True)
        result = df_recommend.merge(top_books,on='book_title')
        result.drop_duplicates('book_title', inplace=True)
        result = result[~result.isbn.isin(top10_isbn)]
        result.reset_index(drop=True, inplace=True)

        ## Render Top 10 Books
        with st.container():
            st.subheader("Top 10 Books")
            for col_index, col in enumerate(st.columns(10)):
                book = top10_books.iloc[col_index]
                title = book["book_title"]
                poster = book['img_url']

                with col:
                    st.image(poster)
                    st.caption(title)

        ## Render Recommended Books
        with st.container():
            st.subheader("You May Also Like These Books")
            topk = 10 if len(result)>=10 else len(result)
            print(">>>",topk)
            for col_index, col in enumerate(st.columns(topk)):
                book = result.iloc[col_index]
                title = book["book_title"]
                poster = book['img_url']
                nvotes = book.NumberOfVotes
                rating = book.AverageRatings
                rec_score = book.rec_score
                with col:
                    st.image(poster)
                    st.caption(title)
                    st.caption(f"{rec_score:.1%} 일치해요!")
                    with st.expander("더보기"):
                        st.caption(f"작가: {book.book_author}")
                        st.caption(f"출판년도: {book.year_of_publication}")
                        st.caption(f"카테고리: {book.category_high}")
                        st.caption(f"요약: {book.summary}")
                        st.caption(f"평가 횟수: {nvotes}")
                        st.caption(f"평균 평점: {rating:.1f}")
                        

        (_, center, _) = st.columns([4, 1, 4])
        with center:
            st.button(
                "Retry",
                on_click=retry,
            )





