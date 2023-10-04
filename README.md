# AI 의류 추천 시스템
- 2021.05.30 ~ 2021.07.31
- 연합 빅데이터 동아리 BITAmin 6기 컨퍼런스
- 6조 박세홍, 이소연, 이지원, 정세영

## Datasets
1) Deepfashion
- part, fabric, shape, texture 4가지 label을 이용해 약 230가지 속성을 분류하는 4개의 CNN 모델링 진행

2) Deepfashion2
- 이미지에서 옷을 detection해주는 mask R-CNN 모델링 진행

3) musinsa
- 위 AI 모델들을 이용해 input으로 들어온 임의의 옷에 대한 코디 output으로 무신사 코디숍 데이터 크롤링해 제공

## Process
![image](https://user-images.githubusercontent.com/58061467/128594630-ec21c99d-50b7-46a8-b652-c56f4b9491fb.png)

## Models
### Mask R-CNN
- matterport에서 제공하는 오픈소스 Mask R-CNN 알고리즘 사용
- coco dataset을 활용해 pre-train 진행
- 이렇게 만든 모델과 가중치에 deepfashion2 데이터셋을 활용해 전이학습 진행
![image](https://user-images.githubusercontent.com/58061467/128594693-e692e04b-60cc-4831-84df-eb8180a9c686.png)

### 4CNNs
![image](https://user-images.githubusercontent.com/58061467/128594703-843be8c8-37bd-491e-857b-5979fae25a98.png)
- deepfashion 데이터셋에서 제공하는 part, shape, fabric, texture 4가지 label 이용해 CNN 모델 학습
- ResNet50의 전이학습을 통해 fastai로 4개의 CNN 훈련

## Recommendation System
- user가 입력한 무신사의 15가지 style 중 하나로 필터링 진행
- 기존 생성된 musinsa data와 input으로 들어온 user 간의 cosine similarity 계산
- 값이 max인 코디와 세부 아이템 출력
