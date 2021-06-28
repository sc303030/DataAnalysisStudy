# 데이콘 - KBO 외국인 투수 스카우팅 최적화 경진대회 [연습]_03

### strike와 ball 분리하기

```python
from tqdm import tqdm

for i in tqdm(range(외국인스탯캐스터.shape[0])):
    if 외국인스탯캐스터.loc[i,'description'].split('_')[-1] in ['strike','foul']:
        외국인스탯캐스터.loc[i,'type'] = 'S'
    elif 외국인스탯캐스터.loc[i,'description'].split('_')[-1] in ['ball']:
        외국인스탯캐스터.loc[i,'type'] = 'B'
    else:
        외국인스탯캐스터.loc[i,'type'] = 'N'

```

![42](./img/42.gif)

- description에 나와있는 설명을 기준으로 strike와 foul은 strike로 ball은 ball로 분리하고 두가지 모두 아니면 출루, 득점, 아웃이기때문에 우선은 제외하였다. rmse가 낮으면 스트라이크로 추가해서 다시 돌려본다. 우선은 제외하고 진행한다.

![43](./img/43.png)

- 이렇게 만든다.

```python
외국인스탯캐스터_index = list(외국인스탯캐스터[외국인스탯캐스터['type'] == 'N'].index)
외국인스탯캐스터_index
외국인스탯캐스터_filter = 외국인스탯캐스터.drop(외국인스탯캐스터_index).reset_index(drop=True)
외국인스탯캐스터_filter
```

- N인 것들은 우선 제외하고 다시 df를 만든다.

### 캐글에 나와있는 스타라이크 존 SVM 해보기[https://www.kaggle.com/jzdsml/sp-finding-baseball-strike-zone-w-svm]

```python
def make_meshgrid(ax, h=.02):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def draw_boundary(ax, clf):

    xx, yy = make_meshgrid(ax)
    return plot_contours(ax, clf, xx, yy,cmap=plt.cm.coolwarm, alpha=0.5)
```

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

def plot_SVM(aaron_judge,gamma=1, C=1):
    aaron_judge.type = aaron_judge.type.map({'S':1, 'B':0})
    fig, ax = plt.subplots()
    plt.scatter(aaron_judge.plate_x, aaron_judge.plate_z, c = aaron_judge.type, cmap = plt.cm.coolwarm, alpha=0.6)
    training_set, validation_set = train_test_split(aaron_judge, random_state=1)
    classifier = SVC(kernel='rbf', gamma = gamma, C = C)
    classifier.fit(training_set[['plate_x', 'plate_z']], training_set['type'])
    draw_boundary(ax, classifier)
    plt.show()
    print("The score of SVM with gamma={0} and C={1} is:".format(gamma, C) )
    print(classifier.score(validation_set[['plate_x', 'plate_z']], validation_set['type']))
```

![44](./img/44.jpg)

- 표본이 적어서 그런지 엄청나게 잘 예측하였다.

#### [svc가 무엇인지, 매게변수가 어떤의미를 가지는지 알아보기](https://bskyvision.com/163)

- 출처 : https://bskyvision.com/163

```
SVC(kernel='rbf', gamma = gamma, C = C)
```

- svm은 데이터를 선형으로 분리하는 최적의 선형 결정 경계를 찾는 알고리즘
- c는 이상치의 정도를 설정한다. c를 높일수록 이상치를 엄격하게 관리하는 것이고 c가 낮을수록 이상치의 허용에 관대하다.
- c가 너무 낮으면 과소적합, c가 너무 높으면 과대적합이 될 가능성이 커진다.

- 선형 svm으로 한계가 있어서 3차원으로 보고 경계를 짓는 rbf 커널 svm이 나왔다.
  - 그래서 커널을 rbf로 지정해주었다.

- 감마가 클수록 작은 표준편차를 가진다.
  - 데이터 포인터가 영향력을 행사하는 거리가 짧아진다.

### 선수별로 SVM은 나중에 구해보고 원래대로 한국에서 경기수와 승률 상위 20%와 하위 20%의 스탯을 뽑아 승패를 학습하기

```python
throw_df = pd.read_csv('률lus_win_lose_2.csv')
top20 = throw_df.sort_values(['승률','경기수'], ascending=False).reset_index(drop=True)[:20]
bottom20 = throw_df.sort_values(['승률','경기수'], ascending=False).reset_index(drop=True)[-20:]
display(top20)
display(bottom20)
```

- 단순히 승률로만 본다면 1경기 뛰고 이기면 100%여서 경기수도 같이 보았다.

![45](./img/45.jpg)

- 상위 20명이여서 그런지 방출도 없다.

- 익숙한 이름들이 많다.

![46](./img/46.jpg)

- 방출된 사람들도 보이고 방어율이 안좋다.

### 머신러닝

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
```

- 사이킷런을 불러서 머신런닝하기

```python
feature = throw_df.drop('승',axis=1)
label = throw_df['승']

X_train, X_test, y_train, y_test = train_test_split(feature,label, test_size = 0.2, random_state=20)
```

- 승을 기준으로 피처와 라벨을 나워서 학습과 테스트 테이터를 나눈다.

```python
base_dtc = DecisionTreeClassifier(random_state=20, criterion='entropy')
base_dtc.fit(X_train, y_train)
predition = base_dtc.predict(X_test)
```

- 우선은 의사결정나무로 학습해보기로 하였다. 

![47](./img/47.jpg)

- 라벨링을 안해서 오류가 발생하였다. 선수이름과 팀, 승을 라벨링해서 다시 나누자.

#### NaN 값 0으로 대체

```python
throw_df['year_born'] = throw_df['year_born'].fillna(0)
throw_df['방출연도'] = throw_df['방출연도'].fillna(0)
```

```python
throw_df.info()
```

```python
from sklearn.preprocessing import LabelEncoder

def label(df):
    col = list(df)
    encoder = LabelEncoder()
    encoder.fit(col)
    col_label = encoder.transform(col)
    return col_label

pitcher_name_label = label(throw_df['pitcher_name'])
team_label = label(throw_df['team'])
year_born_label = label(throw_df['year_born'])
```

- info에서 object인 컬럼들만 라벨인코딩을 실시하였다.

```python
throw_df = throw_df.assign(pitcher_name_label=pitcher_name_label,
                          team_label=team_label,
                          year_born_label=year_born_label)
throw_df_digit = throw_df.drop(['pitcher_name','team','year_born','year'],axis=1)
throw_df_digit.head()
```

- df에 컬럼으로 추가하고 기존에 있던 object컬럼을 제거하였다.

```python
from sklearn.metrics import accuracy_score
print(f'예측 정확도 : {accuracy_score(y_test, predition)}')
>
예측 정확도 : 0.19047619047619047
```

- 다시 예측을 실시하였는데 거의 예측을 안하니만큼 정확도가 나왔다.
- 외국인 스카우트에 있는 정보들을 선수마다 평균을 내서 적용하려고 했지만 한국에서 경기간 횟수가 여러번이면 적용하기 애매해서 우선 참고용으로 쓰기로 하였다.
- year_born은 찾아서 해당 연도에 나이를 컬럼으로 추가하고 방출연도는 방출된 해에 1로 없는해에는 0으로 바꿔서 전체 변수 중요도를 살펴봐야겠다.
- 이렇게해서 찾아보는게 의미가 있는가 싶었다. 그래서 다시 처음으로 돌아가서 방출 당하지 않은 선수들과 방출 당한 선수들의 스탯을 다시 정리해서 파일로 만들고 알고리즘을 짜는게 낫겠다는 생각이 들었다. 그리고 감독이 중요시하는 스탯지들에 가중치를 우선순위로 두어서 다시 정리해보자.
- 다시 보니 2011~2018년도를 학습 데이터로 두고 2019년도를 검증 데이터로 쓰라고 준 것 같다. 그러면 뭔가 학습을 시켜야 하는데 낮게 나와서 의미가 있나 싶다.

- 그럼 승, 패를 두 가지의 선택지라고 두고 의사결정나무로 다시 한 번 머신러닝을 돌려보려고 한다. 방출되지 않은 선수들의 외국 기록을 평균내어서 알고리즘으로 만들고 라벨링을 줘서 그 값을 다시 한국기록데이터에 넣어서 머신러닝을 돌려보려고 한다.

- 계획을 하고 코드를 작성하는게 빠르고 이제 이 미니 프로젝트를 끝내기에 맞는 것 같다.

| 단계 |                             내용                             | 진행/완료 |
| :--: | :----------------------------------------------------------: | :-------: |
|  1   |                투수들의 한국 경기 승, 패 여부                |   완료    |
|  2   | 투수들의 외국 경기 기록 평균으로 변환하여 다시 저장<br>- 기존에 만들어 놓았던 데이터 중에서 구종의 개수를 추가하여 형성 |   진행    |
|  3   |       투수들의 외국 경기 평균 기록에 한국 승패를 추가        |     -     |
|  4   | 1. 한국 기록 데이터 셋<br>2. 외국 기록 데이터 셋으로 각각 의사결정 나무 진행<br>3. 마지막으로 두개의 데이터를 하나로 합쳐서 의사결정나무 진행 |     -     |
|  5   | 위의 결과를 보고 이제 2011~2018데이터를 학습 데이터로 두고 2019 데이터를 검증 데이터로 활용 |     -     |
|  6   | 결과를 보고 마무리 짓고 대회에 입상한 상위 3개팀의 분석 결과를 학습<br>- 부족했던 점과 보완하면 좋은점<br>- 접근 법의 차이와 결과물을 어떻게 시각화 하였는지 등등 비교해보기 |     -     |

### 투수 - 외국 경기 평균 변환

#### 외국인메이저성적에 평균 구속과 구종 개수 추가

```python
외국인메이저성적.head()
```

|      | pitcher_name |   year |  ERA |  WAR |   TBF |     H |   HR |   BB |  HBP |    SO | WHIP | BABIP |  FIP |   LD% |   GB% |   FB% | IFFB% | SwStr% | Swing% |
| ---: | -----------: | -----: | ---: | ---: | ----: | ----: | ---: | ---: | ---: | ----: | ---: | ----: | ---: | ----: | ----: | ----: | ----: | -----: | -----: |
|    0 |       오간도 | 2011.0 | 3.51 |  3.3 | 693.0 | 149.0 | 16.0 | 43.0 |  7.0 | 126.0 | 1.14 | 0.265 | 3.65 | 0.237 | 0.364 | 0.674 | 0.147 |  0.090 |  0.475 |
|    1 |         험버 | 2011.0 | 3.75 |  3.2 | 676.0 | 151.0 | 14.0 | 41.0 |  6.0 | 116.0 | 1.18 | 0.275 | 3.58 | 0.168 | 0.471 | 0.458 | 0.094 |  0.092 |  0.463 |
|    2 |       루카스 | 2012.0 | 3.76 |  2.8 | 827.0 | 185.0 | 13.0 | 78.0 |  1.0 | 140.0 | 1.36 | 0.289 | 3.75 | 0.203 | 0.572 | 0.707 | 0.082 |  0.062 |  0.424 |
|    3 |   다이아몬드 | 2012.0 | 3.54 |  2.2 | 714.0 | 184.0 | 17.0 | 31.0 |  4.0 |  90.0 | 1.24 | 0.292 | 3.94 | 0.210 | 0.534 | 0.597 | 0.040 |  0.068 |  0.467 |
|    4 |     듀브론트 | 2013.0 | 4.32 |  2.2 | 705.0 | 161.0 | 13.0 | 71.0 |  5.0 | 139.0 | 1.43 | 0.310 | 3.78 | 0.199 | 0.456 | 0.633 | 0.127 |  0.077 |  0.434 |

```python
평균구속 = df.groupby('pitcher_name',as_index=False).agg({'km변환':'mean'})\
.sort_values('km변환',ascending=False).reset_index(drop=True)
```

```python
for idx, value in 평균구속.iterrows():
    for i in range(외국인메이저성적.shape[0]):
        if 외국인메이저성적.loc[i,'pitcher_name'] == value[0]:
            외국인메이저성적.loc[i,'평균구속'] = value[1]
```

```python
외국인메이저성적[외국인메이저성적['평균구속'].isna() == True]
```

|      | pitcher_name |   year |   ERA |  WAR |   TBF |    H |   HR |   BB |  HBP |   SO | WHIP | BABIP |  FIP |   LD% |   GB% |   FB% | IFFB% | SwStr% | Swing% | 평균구속 |
| ---: | -----------: | -----: | ----: | ---: | ----: | ---: | ---: | ---: | ---: | ---: | ---: | ----: | ---: | ----: | ----: | ----: | ----: | -----: | -----: | -------: |
|   67 |     벨레스터 | 2010.0 |  2.57 |  0.1 |  89.0 | 15.0 |  2.0 | 11.0 |  2.0 | 28.0 | 1.24 | 0.283 | 3.51 | 0.114 | 0.568 | 0.622 | 0.071 |  0.145 |  0.420 |      NaN |
|  106 |     카스티요 | 2017.0 | 13.50 |  0.0 |   8.0 |  3.0 |  0.0 |  1.0 |  0.0 |  2.0 | 3.00 | 0.600 | 2.41 | 0.600 | 0.400 | 0.500 | 0.000 |  0.200 |  0.467 |      NaN |
|  139 |         리즈 | 2015.0 |  4.24 | -0.2 | 106.0 | 26.0 |  4.0 | 12.0 |  3.0 | 27.0 | 1.63 | 0.367 | 4.98 | 0.349 | 0.349 | 0.690 | 0.158 |  0.103 |  0.461 |      NaN |
|  171 |     벨레스터 | 2015.0 |  7.47 | -0.3 |  77.0 | 17.0 |  3.0 | 13.0 |  1.0 | 13.0 | 1.91 | 0.298 | 6.64 | 0.160 | 0.280 | 0.718 | 0.071 |  0.104 |  0.429 |      NaN |
|  180 |     벨레스터 | 2011.0 |  4.54 | -0.4 | 159.0 | 38.0 |  7.0 | 14.0 |  1.0 | 34.0 | 1.46 | 0.301 | 4.93 | 0.194 | 0.417 | 0.691 | 0.048 |  0.072 |  0.427 |      NaN |
|  190 |     벨레스터 | 2012.0 |  6.50 | -0.5 |  83.0 | 14.0 |  5.0 | 11.0 |  3.0 | 12.0 | 1.39 | 0.173 | 7.71 | 0.161 | 0.321 | 0.684 | 0.069 |  0.077 |  0.418 |      NaN |

- 외국인 메이저 성적 df에 평균 구속 데이터를 추가헀다.
- 거기서 nan값을 가진 정보를 찾아서 스탯 테이터에서도 값이 없으면 지우는 방향으로 가려고 한다. 이걸 또 평균내야 하니 해당 연도를 제외한 평균을 삽입하거나 하려고 한다.

```python
for val in 외국인메이저성적[외국인메이저성적['평균구속'].isna() == True].drop_duplicates(['pitcher_name'])['pitcher_name']:
    display(df[df['pitcher_name'] == val])
```

|      | game_date | release_speed | batter | pitcher | events | description | zone | stand | p_throws | bb_type |  ... |   ax |   ay |   az | launch_speed | launch_angle | release_spin_rate | pitch_name | pitcher_name | km변환 | type |
| ---: | --------: | ------------: | -----: | ------: | -----: | ----------: | ---: | ----: | -------: | ------: | ---: | ---: | ---: | ---: | -----------: | -----------: | ----------------: | ---------: | -----------: | -----: | ---: |
|      |           |               |        |         |        |             |      |       |          |         |      |      |      |      |              |              |                   |            |              |        |      |

0 rows × 26 columns

|      | game_date | release_speed | batter | pitcher | events | description | zone | stand | p_throws | bb_type |  ... |   ax |   ay |   az | launch_speed | launch_angle | release_spin_rate | pitch_name | pitcher_name | km변환 | type |
| ---: | --------: | ------------: | -----: | ------: | -----: | ----------: | ---: | ----: | -------: | ------: | ---: | ---: | ---: | ---: | -----------: | -----------: | ----------------: | ---------: | -----------: | -----: | ---: |
|      |           |               |        |         |        |             |      |       |          |         |      |      |      |      |              |              |                   |            |              |        |      |

0 rows × 26 columns

|      | game_date | release_speed | batter | pitcher | events | description | zone | stand | p_throws | bb_type |  ... |   ax |   ay |   az | launch_speed | launch_angle | release_spin_rate | pitch_name | pitcher_name | km변환 | type |
| ---: | --------: | ------------: | -----: | ------: | -----: | ----------: | ---: | ----: | -------: | ------: | ---: | ---: | ---: | ---: | -----------: | -----------: | ----------------: | ---------: | -----------: | -----: | ---: |
|      |           |               |        |         |        |             |      |       |          |         |      |      |      |      |              |              |                   |            |              |        |      |

0 rows × 26 columns

- 선수들의 정보를 찾아보려고 했는데 없어서 0으로 두고 하려고 한다.

```python
외국인메이저성적_drop_year = 외국인메이저성적.drop('year', axis=1)
```

```python
외국인메이저성적_drop_year.groupby('pitcher_name').agg('mean').head()
```

|              |       ERA |       WAR |        TBF |     H |        HR |        BB |      HBP |        SO |  WHIP |    BABIP |      FIP |      LD% |      GB% |      FB% |    IFFB% |   SwStr% |  Swing% |   평균구속 |
| -----------: | --------: | --------: | ---------: | ----: | --------: | --------: | -------: | --------: | ----: | -------: | -------: | -------: | -------: | -------: | -------: | -------: | ------: | ---------: |
| pitcher_name |           |           |            |       |           |           |          |           |       |          |          |          |          |          |          |          |         |            |
|       니퍼트 |  4.290000 | -0.300000 | 262.000000 |  61.0 |  7.000000 | 34.000000 | 5.000000 | 47.000000 | 1.680 | 0.320000 | 5.090000 | 0.267000 | 0.320000 | 0.666000 | 0.099000 | 0.082000 | 0.42300 | 143.882653 |
|   다이아몬드 | 10.262500 |  0.550000 | 369.500000 | 100.0 | 10.250000 | 21.500000 | 1.250000 | 40.250000 | 2.125 | 0.333750 | 5.660000 | 0.208250 | 0.466250 | 0.641250 | 0.044250 | 0.059000 | 0.44025 | 139.498235 |
|     듀브론트 |  5.106667 |  0.616667 | 377.666667 |  90.0 | 10.500000 | 36.500000 | 2.333333 | 73.666667 | 1.560 | 0.317833 | 4.548333 | 0.205167 | 0.442667 | 0.632167 | 0.104333 | 0.077500 | 0.42850 | 141.691396 |
|       레나도 |  7.306667 | -0.500000 | 131.333333 |  31.0 |  7.333333 | 14.666667 | 0.333333 | 14.666667 | 1.630 | 0.263667 | 6.703333 | 0.173000 | 0.345667 | 0.618000 | 0.101000 | 0.052667 | 0.45400 | 141.203608 |
|         레온 |  6.050000 | -0.100000 |  63.000000 |  16.5 |  2.000000 |  5.000000 | 0.000000 | 10.500000 | 1.585 | 0.303500 | 6.235000 | 0.162500 | 0.479500 | 0.590500 | 0.034500 | 0.101500 | 0.47250 | 140.058685 |

- 선수들의 평균을 내려고 연도를 제거하고 묶었다.
  - 소수점 이하 자리는 다시 조정해서 df로 만들자

```python
외국인메이저_평균 = 외국인메이저성적_drop_year.groupby('pitcher_name').agg('mean').head().reset_index(drop=False).round(decimals=2)
```

- 모두 소수점 2자리까지로 변환

```python
### 구종 개수 추가
playername = list(throw_df['pitcher_name'].drop_duplicates())

구종데이터 = pd.DataFrame({'pitcher_name':[], '2-Seam Fastball':[],'4-Seam Fastball':[],'Changeup':[],'Curveball':[],
                        'Cutter':[],'Intentional Ball':[],'Pitch Out':[],'Slider':[]})
for name in playername:
    try:
        a = 외국인스탯캐스터.groupby(['pitcher_name','pitch_name']).agg({'pitch_name':'count'})\
    .T[name].reset_index(drop=True)
        a['pitcher_name'],a['cnt'] = name, a.shape[1]
        구종데이터 = 구종데이터.append(a,ignore_index=True)
    except:
        print(name)
display(구종데이터.head())
print(구종데이터.cnt.mean())
```

|      | 2-Seam Fastball | 4-Seam Fastball | Changeup | Curveball | Cutter | Eephus | Fastball | Forkball | Intentional Ball | Pitch Out | Sinker | Slider | Split Finger | Unknown |  cnt | pitcher_name |
| ---: | --------------: | --------------: | -------: | --------: | -----: | -----: | -------: | -------: | ---------------: | --------: | -----: | -----: | -----------: | ------: | ---: | -----------: |
|    0 |            26.0 |           384.0 |     81.0 |      82.0 |    6.0 |    NaN |      NaN |      NaN |              4.0 |       1.0 |    NaN |    4.0 |          NaN |     NaN |  8.0 |       니퍼트 |
|    1 |            19.0 |           245.0 |      8.0 |       NaN |   20.0 |    NaN |      NaN |      NaN |              4.0 |       2.0 |  269.0 |  306.0 |          NaN |     NaN |  8.0 |         소사 |
|    2 |            58.0 |           433.0 |    429.0 |       NaN |  171.0 |    NaN |      NaN |      NaN |              9.0 |       5.0 | 1360.0 |  284.0 |          NaN |     NaN |  8.0 |       탈보트 |
|    3 |           773.0 |           950.0 |    297.0 |      88.0 |    NaN |    NaN |      NaN |      NaN |              NaN |       3.0 |    NaN |  256.0 |          NaN |     NaN |  6.0 |     레이예스 |
|    4 |           207.0 |           324.0 |    182.0 |      25.0 |    NaN |    NaN |      NaN |      NaN |              1.0 |       NaN |    NaN |  153.0 |          NaN |     NaN |  6.0 |         세든 |

- 구종을 구한 후 개수로 변환
  - int로 변환하기

```python
투수이름 = 구종데이터['pitcher_name']
구종데이터 = 구종데이터.loc[:,구종데이터.columns != 'pitcher_name'].fillna(0).astype(int)

구종데이터['pitcher_name'] = 투수이름

구종데이터.head()
```

|      | 2-Seam Fastball | 4-Seam Fastball | Changeup | Curveball | Cutter | Eephus | Fastball | Forkball | Intentional Ball | Pitch Out | Sinker | Slider | Split Finger | Unknown |  cnt | pitcher_name |
| ---: | --------------: | --------------: | -------: | --------: | -----: | -----: | -------: | -------: | ---------------: | --------: | -----: | -----: | -----------: | ------: | ---: | -----------: |
|    0 |              26 |             384 |       81 |        82 |      6 |      0 |        0 |        0 |                4 |         1 |      0 |      4 |            0 |       0 |    8 |       니퍼트 |
|    1 |              19 |             245 |        8 |         0 |     20 |      0 |        0 |        0 |                4 |         2 |    269 |    306 |            0 |       0 |    8 |         소사 |
|    2 |              58 |             433 |      429 |         0 |    171 |      0 |        0 |        0 |                9 |         5 |   1360 |    284 |            0 |       0 |    8 |       탈보트 |
|    3 |             773 |             950 |      297 |        88 |      0 |      0 |        0 |        0 |                0 |         3 |      0 |    256 |            0 |       0 |    6 |     레이예스 |
|    4 |             207 |             324 |      182 |        25 |      0 |      0 |        0 |        0 |                1 |         0 |      0 |    153 |            0 |       0 |    6 |         세든 |

