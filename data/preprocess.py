import pandas as pd
import re
import os

def clean_promo(text):
    if not isinstance(text, str):
        return ""
    
    # 링커리어 홍보 문구 등 제거
    # 패턴: "이 글은" 으로 시작하거나 "👉" 문자가 포함되어 있고, "링커리어", "자만검", "만능검색기" 등을 언급하는 앞부분
    text = re.sub(r'^이 글은.*?자(?:만검|소서 만능검색기).*?(?:확인하세요|확인해 보세요|참고해보세요|확인할 수 있습니다)[!\.]\n*', '', text, flags=re.DOTALL)
    text = re.sub(r'^이 글은.*?(?:만능검색기|자만검).*?\n+', '', text, flags=re.DOTALL)
    text = re.sub(r'^이 글은.*?👉.*?\n+', '', text, flags=re.DOTALL)
    
    # 간혹 남는 👉 문장 제거
    text = re.sub(r'👉.*?(?:확인하세요|참고해보세요).*?\n+', '', text)
    
    # 앞에 있는 빈 줄 제거
    text = text.lstrip()
    return text

def parse_qa(text):
    if not text:
        return [{'q': '', 'a': ''}]
        
    # 질문을 구분하는 정규식 패턴
    # 주로 숫자로 시작: '1. ', '[1]', '1-1.', 'Q.', '[질문]' 등
    pattern = r'\n(?=(?:\[?\d+(?:-\d+)*\]?[\.\)\]]\s|\d+[\.\)]\s|Q\d*\.?\s|▼\s|\[질문\]|※\s?자기소개서|■\s))'
    
    text = '\n' + text.strip()
    pieces = re.split(pattern, text)
    
    qa_pairs = []
    
    # 질문 패턴 없이 시작하는 경우를 위해
    # 첫번째 piece가 내용만 있다면 질문 없이 답변만 있는 것으로 간주할 수도 있음
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
            
        parts = piece.split('\n', 1)
        q = parts[0].strip()
        a = parts[1].strip() if len(parts) > 1 else ""
        
        # 질문인지 판별 (숫자로 시작하거나, 너무 짧은 문장이면 질문)
        # 만약 전체가 한 문단이면 전체를 a로 넣음
        is_question_format = re.match(r'^(?:\[?\d+(?:-\d+)*\]?[\.\)\]]\s|\d+[\.\)]\s|Q\d*\.?\s|▼\s|\[질문\]|※\s?자기소개서|■\s)', q)
        
        if is_question_format:
            qa_pairs.append({'q': q, 'a': a})
        else:
            # 질문 형태가 아닐 경우 이전 답변에 합치거나, 첫번째 요소면 질문 빈칸, 텍스트 전체를 답변으로 처리
            if not qa_pairs:
                qa_pairs.append({'q': '', 'a': piece})
            else:
                qa_pairs[-1]['a'] += '\n\n' + piece
                
    if not qa_pairs:
        qa_pairs.append({'q': '', 'a': text.strip()})
        
    return qa_pairs

def main():
    input_file = 'linkareer_it_cover_letters.csv'
    output_file = 'linkareer_it_cover_letters_qa.csv'
    
    df = pd.read_csv(input_file)
    
    processed_rows = []
    
    for _, row in df.iterrows():
        content = row.get('자기소개서_내용', '')
        cleaned_content = clean_promo(content)
        qa_pairs = parse_qa(cleaned_content)
        
        for qa in qa_pairs:
            # 원본 행의 데이터 복사
            new_row = row.to_dict()
            del new_row['자기소개서_내용'] # 기존 column 제거
            
            new_row['질문'] = qa['q']
            new_row['답변'] = qa['a']
            
            processed_rows.append(new_row)
            
    # 새 데이터프레임 생성
    # 컬럼 순서 지정: id,회사명,직무,합격시기,지원형태_기업구분,합격스펙,스크랩수,질문,답변,URL
    columns = ['id', '회사명', '직무', '합격시기', '지원형태_기업구분', '합격스펙', '스크랩수', '질문', '답변', 'URL']
    out_df = pd.DataFrame(processed_rows)
    
    # 존재하는 컬럼만 정렬
    existing_cols = [col for col in columns if col in out_df.columns]
    out_df = out_df[existing_cols]
    
    # 필드가 완전히 비어있는 행(질문, 답변 둘다 없는 경우) 정리
    out_df = out_df.dropna(subset=['답변'])
    out_df = out_df[out_df['답변'].str.strip() != '']
    
    out_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"전처리 완료. 총 {len(out_df)}개의 질문-답변 세트 추출됨.")
    print(f"저장 위치: {os.path.abspath(output_file)}")

if __name__ == '__main__':
    main()
