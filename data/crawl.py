import urllib.request
import json
import re
import pandas as pd
import time
import datetime
import os

def fetch_page(page):
    url = f'https://linkareer.com/cover-letter/search?page={page}&role=IT&sort=RECENT_SCRAP_COUNT'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        html = urllib.request.urlopen(req).read().decode('utf-8')
        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
        if match:
            data = json.loads(match.group(1))
            apollo_state = data.get('props', {}).get('pageProps', {}).get('__APOLLO_STATE__', {})
            cover_letters = [v for k, v in apollo_state.items() if 'CoverLetter:' in k]
            return cover_letters
    except Exception as e:
        print(f"Error fetching page {page}: {e}")
    return []

def main():
    all_cover_letters = []
    
    # IT 직무, 스크랩순 1페이지~20페이지 수집
    for page in range(1, 21):
        print(f"Fetching page {page}...")
        letters = fetch_page(page)
        if not letters:
            print("No more cover letters found or error occurred. Stopping.")
            break
            
        all_cover_letters.extend(letters)
        time.sleep(1) # 서버 부하 방지
        
    print(f"총 {len(all_cover_letters)}건의 데이터를 수집했습니다.")
    
    # 데이터 가공
    dataset = []
    for item in all_cover_letters:
        passed_at = item.get('passedAt')
        passed_date = ""
        passed_year = ""
        
        if passed_at:
            try:
                dt = datetime.datetime.fromtimestamp(passed_at / 1000.0)
                passed_year_str = str(dt.year)
                month = dt.month
                half = "상반기" if month <= 6 else "하반기"
                passed_date = f"{passed_year_str} {half}"
            except:
                pass
                
        # 합격 스펙 문자열로 결합
        specs = []
        for key in ['university', 'major', 'grades', 'languageScore', 'career', 'activity', 'license']:
            val = item.get(key)
            if val:
                specs.append(str(val))
        
        # 기업명 처리 등
        row = {
            'id': item.get('id'),
            '회사명': item.get('organizationName'),
            '직무': item.get('role'),
            '합격시기': passed_date,
            '지원형태_기업구분': ", ".join(item.get('types', [])) if isinstance(item.get('types'), list) else item.get('types'),
            '합격스펙': " / ".join(specs),
            '스크랩수': item.get('scrapCount', 0),
            '자기소개서_내용': item.get('content', ''),
            'URL': f"https://linkareer.com/cover-letter/{item.get('id')}" if item.get('id') else ''
        }
        dataset.append(row)
        
    df = pd.DataFrame(dataset)
    # 중복 제거 (GraphQL apollo state 특성상 중복 ID가 존재할 수 있음)
    df = df.drop_duplicates(subset=['id'], keep='first')
    
    output_dir = 'c:/Users/simyo/Desktop/jasoser_AI-main/data'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'linkareer_it_cover_letters.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig') # MS Excel에서 한글 깨짐 방지
    print(f"중복 제거 후 최종 {len(df)}건의 자소서 데이터가 {output_path}에 저장되었습니다.")

if __name__ == '__main__':
    main()
