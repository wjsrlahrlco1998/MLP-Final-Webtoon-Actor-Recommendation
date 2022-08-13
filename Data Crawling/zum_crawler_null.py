import pandas as pd
data = pd.read_excel('zum_data_test3.xlsx')
data
name= data['Name']
import time
from random import randint
headers= {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
}
from tqdm import tqdm
import requests
import time
from bs4 import BeautifulSoup
num_list = ['0']*1878
log_df = pd.DataFrame({'Name':data['Name'],'null_actor_list':num_list})

for k in tqdm(range(0, 30)):
    actor_info = []
    log = []            
    if len(str(data['커뮤니티'][k])) <= 30:
        for i in range(1, 21):        
            try:
                query = '배우'+' '+data['Name'][k]
                resp = requests.get(f'https://search.zum.com/search.zum?method=board&option=accu&query={query}&rd=1&startdate=&enddate=&datetype=&scp=0&page={i}&mm=direct',headers=headers)
                soup = BeautifulSoup(resp.text)
                title = soup.select('.tit')[5:]
                content = soup.select('.txt_wrap')
                for j in range(0,10):
                    actor_info.append(title[j].text + content[j].text)
                print(f'{k}번째 배우 {i}페이지 크롤링 중')            
                time.sleep(randint(1,3))
                
            except:
                log.append(k)
                print(f'{k}번째 배우 오류')
                time.sleep(randint(1,3))
    else:
        print(f'{k}번째 배우 데이터 충분')

    log_df['null_actor_list'].iloc[k] = log
    data['커뮤니티'].iloc[k] = actor_info
    data.to_excel('zum_data_null.xlsx')
    log_df.to_excel('zum_log.xlsx')