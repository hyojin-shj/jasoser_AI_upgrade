import pandas as pd
import json

def main():
    csv_file = 'linkareer_it_cover_letters_qa.csv'
    json_file = 'linkareer_it_cover_letters.json'
    
    df = pd.read_csv(csv_file)
    
    # Fill NaN values with empty string
    df = df.fillna('')
    
    results = []
    
    # ID 단위로 그룹화하여 자소서별로 묶기
    grouped = df.groupby('id', sort=False)
    
    for uid, group in grouped:
        first_row = group.iloc[0]
        
        # 기본 정보 세팅
        item = {
            'id': int(uid) if isinstance(uid, (int, float)) and not isinstance(uid, str) else str(uid),
            '회사명': first_row['회사명'],
            '직무': first_row['직무'],
            '합격시기': first_row['합격시기'],
            '지원형태_기업구분': first_row['지원형태_기업구분'],
            '합격스펙': first_row['합격스펙'],
            '스크랩수': int(first_row['스크랩수']) if str(first_row['스크랩수']).isdigit() else first_row['스크랩수'],
            'URL': first_row['URL']
        }
        
        # 질문과 답변을 순서대로 할당 (question1, answer1, question2, answer2 등)
        idx = 1
        for _, row in group.iterrows():
            q = row['질문']
            a = row['답변']
            item[f'question{idx}'] = q.strip()
            item[f'answer{idx}'] = a.strip()
            idx += 1
            
        results.append(item)
        
    # JSON 파일로 저장
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"변환 완료! 총 {len(results)}개의 자소서가 JSON으로 저장되었습니다.")

if __name__ == '__main__':
    main()
