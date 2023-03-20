## 모델 개요

- FM : SVM 과 Factorization Model의 장점을 결합한 모델
- FFM : 입력 변수를 필드(field)로 나누어, 필드별로 서로 다른 latent factor를 가지도록 factorize
- NCF : Neural Net 기반의 architecture인 Neural Collaborative Filtering
- WDN : 선형적인 모델과 비선형적인 모델을 결합한 모델
- DCN : Cross Network를 통해 feature를 교차 학습함으로 써, 여러 feature 간 특성이 같이 고려된 추천을 하는 모델
- CNN_FM : 시각적 선택을 고려하기 위한 CNN모델과 Feature interaction을 고려할 수 있는 FM 모델을 결합한 모델
- DeepCoNN : User와 Item의 Text 컨텐츠를 2개의 병렬 TextCNN 구조를 통해 User와 Item간의 임베딩을 학습하는 모델

<br>

## 모델 선정 및 분석
1. DeepCoNN, CNN_FM 모델 선정 및 분석

   EDA를 통해 평점 데이터가 99.996% Sparse한 데이터셋인 것을 확인하였습니다. 이런 Cold Start 문제를 활용하기 위해 텍스트나 이미지와 같은 비정형 데이터를 활용하여 성능 개선을 하고자 해당 모델을 사용하였습니다.

2. FFM 모델 선정 및 분석

   책에 대한 텍스트 데이터와 이미지 데이터가 한정적임으로 인해, DeepCoNN과 CNN_FM을 통한 Cold Start 개선이 어느정도 이상 되지 않았습니다. 이 문제를 해결하기 위해, 비교적 다양한 피처들이 존재했던 User에 대한 Context 정보와 Books에 대한 Context 정보를 활용하는 것 이었습니다. 그래서 FFM 모델을 사용하였습니다.

  <br>

## 모델 평가 및 개선
 1. DeepCoNN 모델 평가 및 개선

     <img width="400" alt="image" src="https://user-images.githubusercontent.com/71438046/226345709-eb91d029-308d-4a14-8308-fbfac13d173b.png">

     Books의 Summary Text 데이터를 메인으로 사용하는 Text 기반 모델입니다. 2개의 병렬 TextCNN 구조를 활용하여 하나는 User가 읽은 책에 대한 Summary 정보, 다른 하나는 책 자체의 Summary 정보를 활용하였습니다. Summary 정보를 books의 title로 채워보는 등 다양한 데이터 프로세싱으로 성능을 개선시켜 보았습니다. 하지만 데이터를 채우는거로는 큰 성능 개선이 되지 않았고, EMBED_DIM과 LATENT_DIM 등의 파라미터를 활용한 성능 개선의 폭이 더 컸습니다.
     
 2. CNN_FM 모델 평가 및 개선
     
     책의 표지 이미지로 부터 특징을 추출해 Cold Start 문제를 완화할 수 있는 이미지 기반 모델로, 이미지 처리를 하는 CNN 모델과 Latent Factor를 통해 Feature interaction을 고려할 수 있는 FM 모델을 결합한 모델입니다. 이미지 표지 컬럼의 경로에는 결측치가 없었지만, 이미지 자체가 보라색으로 채워져있거나 하는 이상 데이터들이 꽤 많았습니다. 이미지 데이터를 추가로 스크래핑하는것은 허용되지 않아서, 역시 하이퍼 파라미터를 조정하며 성능 개선을 해보았습니다. 
     
 3. FFM 모델 성능 개선

    FFM모델은 확실히 FM 모델보다 성능이 좋았습니다. FM은 모든 Iatent vector간 interaction을 하나의 벡터로 표현하기 때문에 생기는 한계를, 필드마다 latent vector를 만들어 개선했기 때문입니다. Users에 존재하는 location, age 정보와 books의 category, publisher, language등의 피처들을 다양한 방식으로 processing 하며 성능을 개선시켰습니다. 

  4. DCN / NCF 모델 성능 개선
     
     두 모델 모두 조치에 피쳐로 ‘user_id’와 ‘isbn’만을 가지고 동작하였습니다. 추가적으로 피처를 더해준다면 성능의 개선이 이루어 질수 있다고 판단하여 다양한 피처를 추가하고 결과를 비교하는 방식으로 가설을 검증했습니다. 이때 2개 이상의 피처를 추가할 경우 성능 향상은 이루어 지지만 하나의 피처만을 추가했을때 보다 상승폭이 낮아서 하나의 피쳐만을 사용하도록 진행했습니다. 두개의 모델 모두 location을 추가했을때 대락 2.8% 성능 향상이 이루어졌습니다.
     
  5. WDN 모델 성능 개선
     
     context model과 같이 feature를 모두 적용하였습니다. overfitting이 쉽게일어나는듯하여 Dropout 값을 Default 0.1 → 0.5로 높여주어 최적값을 찾았습니다. MLP_LAYER는 16의 값으로 여러개 쌓는 값이 가장 잘 나왔습니다. EMB_DIM은 점점 높여가며 val_loss를 비교한 결과 32에서 최적의 결과가 나오는 것을 확인하였습니다.

