# AI Team Project

## 목표

실시간 음성을 입력을 받아 음계를 예측하고 예측한 음계에 맞는 캐릭터 움직이는 프로젝트

<br>

----------------

<br>


## preprocessing 모듈

**학습시킬 데이터를 전처리 하여 npy 파일로 저장** 

### 전처리 과정


<details>
<summary>
펼치기/접기
</summary>


1. 학습 시킬 원본데이터 분석

- 학습 시킬 wav 파일을 matplolib으로 출력하니 x축은 시간 y축은 데시벨이 출력돰
  ![db_graph](https://user-images.githubusercontent.com/50133267/101586871-30fd3180-3a26-11eb-9048-7150dd0c24a1.png)

- 실시간 음계를 찾아내기 위해서 x축의 시간축을 제거하고 주파수로 나타내는 그래프를 만듬
  ![FQ_Graph](https://user-images.githubusercontent.com/50133267/101587256-21321d00-3a27-11eb-9dd3-002afa314dbd.png)

    - 그래프 중 해당 음계에 맞지 않는 주파수가 높게 나타나는 상황이 발생 -> Scatter로 확인.
      ![SQ_SCAtter](https://user-images.githubusercontent.com/50133267/101587415-838b1d80-3a27-11eb-8e58-2ecc9c8a34e7.png)

    - 해당 주파수를 midi 번호로 변경 한 뒤 round를 통해 그룹화 진행, 그룹화 데이터의 평균을 구하면 원하는 midi 번호가 높은 power를 가지는 그래프를 볼 수 있었음.
      ![midi_graph](https://user-images.githubusercontent.com/50133267/101587765-3ce9f300-3a28-11eb-92a8-2836254cfbb6.png)

    - 주파수에 해당하는 미디번호

    ![KakaoTalk_20201204_093038183](https://user-images.githubusercontent.com/50133267/101587779-42dfd400-3a28-11eb-9a50-99e52f2e2fad.png)

- 만들어진 그래프의 powr를 x값으로, wav 파일에 명시되어 있는 midi번호를 y값으로 npy 저장

</details>
<br>

----

<br>

## model 모듈

**전처리 한 npy 파일을 이용하여 모델을 학습하여 모델을 저장**

<details>


<summary>
펼치기/접기
</summary>

1. model의 빠른 생성을 위해 lgbm 사용

- lgbm의 특성
  - 적은 메모리 사용
  - 높은정확도
  - GPU 사용
  - 데이터의 1만개 이하의 경우 overfitting 위험 

- lgbm 사용 시 acc: 0.86의 결과를 보여줌

```
model = LGBMClassifier(n_jobs=-1,
                     tree_method='gpu_hist',
                     predictor = 'gpu_predictor'
                     )
```

</details>
<br>

----

<br>

## live_to_scale

**저장한 모듈을 사용하여 실시간 마이크 입력으로 도, 레, 미, 파, 솔, 라, 시 출력**

<details>
<summary>
펼치기/접기
</summary>


1. 실시간 데이터 입력받기 
   - pyaudio를 이용하여 실시간 stram받기
   - stream을 frombuffer를 사용하여 바이너리에서 float으로 추출
2. 데이터 전처리
   - 추출한 데이터를 학습시킨 전처리과정과 동일한 전처리 
3. 결과출력
   - lgbm 모델을 load, 전처리가 끝난 실시간 데이터를 사용하여 predict를 추출
   - predict는 midi 번호, 해당 미디번호에 맞는 음계(도, 레 ...)를 출력 

</details>

