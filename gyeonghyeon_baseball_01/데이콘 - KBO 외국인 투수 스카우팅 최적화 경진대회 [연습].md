# 데이콘 - KBO 외국인 투수 스카우팅 최적화 경진대회 [연습]

### 데이터 불러오기

```python
import pandas as pd
```

```python
외국인역대성적 = pd.read_csv('kbo_yearly_foreigners_2011_2018.csv')
외국인메이저성적 = pd.read_csv('fangraphs_foreigners_2011_2018.csv')
외국인스탯캐스터 = pd.read_csv('baseball_savant_foreigners_2011_2018.csv')
신규외국인성적 = pd.read_csv('fangraphs_foreigners_2019.csv')
신규외국인스텟캐스터 = pd.read_csv('baseball_savant_foreigners_2019.csv')
```

### 파일 확인

```python
display(외국인역대성적.head())
print('*'*50)
display(외국인메이저성적.head())
print('*'*50)
display(외국인스탯캐스터.head())
print('*'*50)
display(신규외국인성적.head())
print('*'*50)
display(신규외국인스텟캐스터.head())
print('*'*50)
```

![01](./img/01.jpg)

- 다음과 같이 파일이 어떤 구성으로 되어있는지 확인하였다.

### 승리와 패배 크롤링 하기

- 승리와 패배가 없어서 [기록실](https://www.koreabaseball.com/Record/Player/PitcherBasic/Basic1.aspx)에서 크롤링하기로 하였다.

```python
name = list(set(외국인역대성적['pitcher_name']))

name_dict = {}
for data in name:
    name_dict[data] = 0
```

- 우선 외국인 선수의 이름을 담은 리스트를 만들어서 각자 값을 담을 딕셔너리를 만든다.

#### 패키지 import

```python
from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.error   import HTTPError
from urllib.error   import URLError

from selenium import webdriver
import time
path = '../driver/chromedriver.exe'
driver = webdriver.Chrome(path)
```

#### 크롤링 함수 

```python
driver.get('https://www.koreabaseball.com/Record/Player/PitcherBasic/Basic1.aspx')
page1 = driver.find_elements_by_tag_name('option')
for year in [ i for i in range(2011,2019)]:
    page1 = driver.find_elements_by_tag_name('option')
    for i in page1:
        if i.text == str(year):
            print(year)
            i.click()
            time.sleep(1.5)
            page = driver.find_elements_by_css_selector('.tData01')
            k = page[0].find_element_by_tag_name('thead').find_element_by_tag_name('tr').find_elements_by_tag_name('th')
            k[5].click()
            time.sleep(1.5)
            page3 = driver.find_elements_by_css_selector('.paging')
            for page_num in range(1,6):
                page3 = driver.find_elements_by_css_selector('.paging')
                page3[0].find_elements_by_tag_name('a')[page_num].click()
                time.sleep(1.5)
                page2 = driver.find_elements_by_css_selector('.tData01')
                a = page2[0].find_element_by_tag_name('tbody').find_elements_by_tag_name('tr')
                for idx,value in enumerate(a):
                    print(value.find_elements_by_tag_name('td')[1].text)
```

![02](./img/02.png)

- 우선 연도를 클릭해야 해서 해당 태그를 찾아 option이랑 루프 구문이 같으면 클릭`i.click()`해서 해당 연도로 넘어가도록 하였다.

![03](./img/03.png)

- 그 다음에 처음 화면은 제한된 선수만 보여줘서 W`k[5].click()`를 클릭하여 모든 선수를 볼 수 있도록 한다.

![04](./img/04.png)

- 다음과 같이 정렬하도록 click()속성을 준다.

![05](./img/05.png)

- 총 5페이지여서 `page3[0].find_elements_by_tag_name('a')[page_num].click()` 하여 페이지를 넘긴다.
- 다시 2012년도로 돌아갈 때 다음과 에러가 발생하였다.

![06](./img/06.jpg)

- 그래서 함수로 만들어서 년도별로 저장하려고 한다.