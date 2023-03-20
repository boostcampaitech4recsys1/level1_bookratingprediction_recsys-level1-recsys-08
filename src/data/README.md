# 탐색적 분석 및 전처리


## A. 데이터 소개
<img width="851" alt="Untitled (2)" src="https://user-images.githubusercontent.com/79351899/226344839-1d4ade91-4ef5-4947-a4d9-b80979e2e71b.png">

- 학습 데이터
    - users.csv에는 68,092명의 고객 정보가 담겨있습니다.
    - books.csv에는 149,580개의 책 정보가 담겨있습니다.
    - train_ratings.csv에는 306,795건의 평가 정보가 담겨있습니다.

- 테스트 데이터
    - 테스트 데이터는 학습 데이터의 train_ratings.csv와 동일한 column을 가지고 있고, rating은 0으로 채워져있습니다.
    - 테스트 데이터는 76,699건의 예측해야하는 user_id와 isbn조합이 담겨있습니다.

## B. 데이터 분석 및 Feature Engineering

- 데이터 전처리
    - 전체  
        - (indexing) 모든 rating을 제외한 모든 column값을 인덱싱 처리해주었습니다.
        
    - users
        - (fill in) age의 결측값에 대해선 평균을 취해주었습니다.
        - (split & fill in) 쉼표로만 구분되었던 ocation정보를 country, state, city로 나눈 후 city를 기준으로 country와 state의 결측값을 채워주었습니다.
        
    - books
        
        - (refine) : 동일한 publisher의 다른 이름을 동일하게 해주었습니다.
        
        - (bounding) : 카테고리들을 상위 카테고리로 묶어주었습니다.
        
- 데이터 분석
    - sparsity
        
        - 최대가능한 평점 기록 개수 : 유저수 * 책수 = 68069 * 149570 = 10181080330
        
        - 주어진 평점 기록 개수 : 306,795
        
        - sparsity : 1 - 주어진 평점 기록 개수 / 최대 가능한 평점 기록 개수 = 99.996%
        
    - train 데이터와 test 데이터의 분포
        
        - Adversarial Validation의 방법을 통해 train 데이터와 test 데이터의 데이터분포의 유사도를 확인해보았습니다.
        
        - 'user_id' / 'isbn'/ 둘다 인 세가지의 경우에 비교를 해 본 결과 모두 0.5에 비슷한 분포를 가지고 있다는 것을 확인할 수 있었습니다.
