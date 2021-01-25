# 데이콘 - KBO 외국인 투수 스카우팅 최적화 경진대회 [연습]_02

### 만들었던 csv 불러오기

```python
외국인역대성적 = pd.read_csv('kbo_yearly_foreigners_2011_2018_추가분.csv',encoding='utf-8-sig',engine='python',sep=',')
외국인역대성적.head()
```

- 지난번에 승_패를 저장했던 파일을 다시 불러온다.

![16](./img/16.jpg)

- 그러면 이렇게 정상적으로 불러와진다. 현재 승,패가 같이 붙어있으니 따로 나눠서 승,패로 새로운 컬럼으로 넣어주자.

```python
외국인역대성적['승'] = 외국인역대성적['2011년_승_패'].apply(lambda x:str(x).split(' ')[0])
외국인역대성적['패'] = 외국인역대성적['2011년_승_패'].apply(lambda x:str(x).split(' ')[1])

외국인역대성적.drop('2011년_승_패',axis=1,inplace=True)
```

- 공백으로 나눠서 0번째는 승, 1번째가 패이다. 그 다음에 원래있던 2011\_승_패 컬럼을 제거했다.

- 그러면 승률과 경기수를 추가하자.

```python
def winratio(x):
    return round(int(x['승']) / (int(x['승'])+int(x['패'])),3)

외국인역대성적['승률'] = 외국인역대성적.apply(lambda x:winratio(x),axis=1)
외국인역대성적['경기수'] =  외국인역대성적.apply(lambda x:int(x['승']) + int(x['패']),axis=1)
```

- 이렇게 하면 승률과 관련된 정보들이 모였다.

![17](./img/17.jpg)

#### 승률과 경기수로 상위 10명, 하위 10명 알아보기

```python
def headtail10(df,text):
    display(df.sort_values(text,ascending=False).head(10))
    print('*'*20,'head','*'*20)
    display(df.sort_values(text,ascending=False).tail(10))
```

```python
headtail10(외국인역대성적,['경기수','승률'])
```

![18](./img/18.jpg) 

- 경기수가 많다는건 팀에서 핵심 투수라는 건데 경기수가 낮은 투수들이 있었다. 경기수도 많으면서 승률이 높은 투수들의 특징이 무엇인지 분석하면 될 것 같다.

```python
headtail10(외국인역대성적,['승률','경기수'])
```

![19](./img/19.jpg)

- 상위권은 위랑 비슷하지만 하위권이 다르다. 경기수는 많지만 승률이 낮은 투수들이 있었다. 이것도 비교하면 좋겠다.

```python
headtail10(외국인역대성적,['경기수'])
```

![20](./img/20.jpg)

- 경기수가 적은 선수들이 과연 왜그럴까 생각해봤더니 방출가능성이 있었다. 그레서 방출된 선수들의 명단을 찾아서 정보를 날리기로 하였다. 방출된 선수들의 정보도 따로 보려고 한다.

#### 방출 선수 찾기

- [외국인선수목록]([https://namu.wiki/w/%EC%99%B8%EA%B5%AD%EC%9D%B8%20%EC%84%A0%EC%88%98/%EC%97%AD%EB%8C%80%20%ED%94%84%EB%A1%9C%EC%95%BC%EA%B5%AC](https://namu.wiki/w/외국인 선수/역대 프로야구))
- 여기서 그냥 찾아서 엑셀에 기록하였다.

![21](./img/21.png)

### 방출선수들 세부기록 보기

```python
throw_df = pd.read_csv('plus_win_lose_2.csv')
throw_df.head()
```

- 위에서 입력한 파일을 열었다.

```python
throw_df_01 =  throw_df[throw_df['방출연도'].isnull() == False].reset_index(drop=True)
throw_df_01['방출연도'] = throw_df_01['방출연도'].astype(int)
```

- 방출연도라는 새로운 컬럼을 위에서 만들었고 그 값이 null이 아닌 것들만 추려서 인덱스와 타입을 바꾸었다.

```python
throw_df_01.describe()
```

![22](./img/22.jpg)

- 패한 경기수가 더 많다.

- era보 6.x대라 낮다고 볼 수 없다. 그러면 게임수로 나눠보자.

```python
throw_df_01[throw_df_01['경기수'] >= 10].describe()
```

![23](./img/23.jpg)

- 10경기 이상 뛴 선수로 나눈 것은 어느정도 괜찮은 성적을 내고있다고 생각해서 그만큼 경기를 뛰었다고 생각했기때문이다.
- 기대만큼 다들 못해줘서 방출이라는 결과가 나온것 같다. 세부기록들은 다른 방출 안 된 선수들과 비교해보자.

```python
throw_df_01[throw_df_01['경기수'] < 10].describe()
```

![24](./img/24.jpg)

- 승률도 낮고 평균자책점도 높다. 

```python
throw_df[throw_df['pitcher_name'] == '마야']
```

![24](./img/25.jpg)

- 이렇게 2년간 뛰다가 방출된 선수들도 있어서 비고해보기로 하였다. 확실히 2015년에 기록이 안좋아졌다.