import re
import requests
import os
import urllib.request
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import json
import unicodedata

load_dotenv()

client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

def get_parser(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 200 아니면 HTTPError 발생
        return BeautifulSoup(response.text, "html.parser")

    except requests.exceptions.HTTPError as e:
        print(f"[HTTP 오류] 상태코드: {e.response.status_code}, 사유: {e.response.reason}")

    except requests.exceptions.ConnectionError as e:
        print("[네트워크 오류] 서버에 연결할 수 없음:", e)

    except requests.exceptions.Timeout:
        print("[타임아웃 오류] 서버 응답 지연")

    except requests.exceptions.RequestException as e:
        print("[알 수 없는 오류]", e)

    return None


def get_mobile_naver_content(url):

    url = url.replace("blog.naver.com", "m.blog.naver.com")

    bs = get_parser(url)

    if bs == None:
        return "제목 없음", "내용 없음"

    title_tag = (
        bs.find("h3", class_="se_textarea") or
        bs.find("div", class_="se-title-text") or
        bs.find("div", id="title") or
        bs.find("strong") or
        bs.find("title")  # 최후 fallback: HTML <title> 태그
    )
    title = title_tag.get_text(strip=True) if title_tag else "제목 없음"

    texts = None

    content = bs.find("div", class_="se-main-container")
    if content:
        texts = content.get_text(" ", strip=True)

    if not texts:
        wraps = bs.find_all("div", class_="se_component_wrap")
        if wraps:
            texts = " ".join([w.get_text(" ", strip=True) for w in wraps])

    if not texts:
        legacy = (
            bs.find("div", id="postViewArea") or
            bs.find("div", class_="post-view") or
            bs.find("div", id="viewTypeSelector")
        )
        if legacy:
            texts = legacy.get_text(" ", strip=True)

    if not texts:
        return title, "내용 없음"

    clean_text = re.sub(r'\s+', ' ', texts)
    return title, clean_text

def crawl_data_to_json(keyword_path, json_path='json_data.json'):
    with open(keyword_path, "r", encoding='utf-8') as f:
        lines = f.readlines()

    common_keyword = [line.strip().split('. ', 1)[-1] for line in lines if line.strip()]
    # print(common_keyword)

    # results = {}
    contents = []
    seen_links = set()

    for kw in common_keyword:
        encText = urllib.parse.quote(kw)
        url = f"https://openapi.naver.com/v1/search/blog.json?query={encText}&display=10&start=1"
        request = urllib.request.Request(url)
        request.add_header("X-Naver-Client-Id",client_id)
        request.add_header("X-Naver-Client-Secret",client_secret)
        request.add_header("User-Agent", 'Mozilla/5.0 (Windows NT 10.0;Win64; x64)\
                            AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98\
                            Safari/537.36')
        response = urllib.request.urlopen(request)
        rescode = response.getcode()

        if(rescode==200):
            response_body = response.read()
            result = json.loads(response_body.decode('utf-8'))

            for item in result["items"]:
                link = item.get('link')
                if not link or link in seen_links:
                    continue
                blog_title, blog_content = get_mobile_naver_content(link)
                if blog_content != "내용 없음" and blog_content != "":
                    if '쿨톤' in kw:
                        keyword = '쿨톤'
                    elif '웜톤' in kw:
                        keyword = '웜톤'
                    elif kw.split()[1] in ['봄', '여름', '가을', '겨울']:
                        keyword = '계절별 헤어스타일'
                    elif '특징' in kw:
                        keyword = '머리별 특징'
                    else:
                        keyword = '얼굴형별 헤어스타일'

                    content = {'title': blog_title, 'content': blog_content, 'link': link, 'keyword': keyword}
                    contents.append(content)
                    seen_links.add(link)

                    # results[kw] = contents
                    # 중복 링크 처리 방법 강구
            
            print(f"In {kw}. 총 {len(contents)} 개 결과 존재")

        else:
            print("In" + kw + "Error Code:" + rescode)
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(contents, f, ensure_ascii=False, indent=4)

def clean_doc(s, kin=False, is_title=False):
    if not s:
        return ""
    
    # 1. 유니코드 정규화 (호환 문자 통합)
    s = unicodedata.normalize("NFKC", str(s))
    
    # 2. 제어문자 제거 (Zero-width, 방향 제어 등)
    s = re.sub(r'[\u200b-\u200f\u202a-\u202e\x00-\x1F\x7F]', '', s)
    
    # 3. 전화번호 제거
    s = re.sub(r"\b\d{2,4}-\d{3,4}-\d{4}\b", " ", s)   # 010-1234-5678
    s = re.sub(r"\b\d{9,11}\b", " ", s)                # 01012345678
    s = re.sub(r"\+?\d{1,3}-\d{1,4}-\d{3,4}-\d{4}", " ", s)  # 국제번호
    s = re.sub(r"\(?0\d{1,2}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}", " ", s) # (031)395-5182, 031)395-5182, 031 395 5182

    # 4. 주소 제거
    s = re.sub(r"[가-힣]+(로|길|번길)\s?\d+(-\d+)?", " ", s)
    s = re.sub(r"[가-힣]+(시|군|구|읍|면|동)\s?[가-힣0-9]*", " ", s)
    
    # 5. URL 제거
    s = re.sub(r"http[s]?://\S+|www\.\S+|blog\.naver\.com\S*", " ", s)
    
    # 6. 해시태그 제거
    s = re.sub(r"#\S+", " ", s)

    # 7. 알파벳, 한글, 숫자, 공백, 괄호, 대괄호, 콜론, 하이픈만 남기고 나머지는 제거
    s = re.sub(r"[^0-9a-zA-Z가-힣\s\(\)\[\]\.: \-]+", " ", s)

    # 8. 공백 정리
    s = re.sub(r"\s+", " ", s).strip()

    if is_title or not kin:
        return s

    else:
        sentences = re.split(r'(?<=[.?!])\s+', s)
        # 제거할 광고문구
        remove_phrases = ["맡겨주세요", "감사합니다", "노력하겠습니다", "함께 하세요", "상담", "문의" ]
        
        # 마지막 문장 + 특정 문구 포함 문장 제거
        filtered_sentences = []
        for i, sent in enumerate(sentences):
            if i == len(sentences) - 1:  # 마지막 문장 제거
                    continue
            if any(phrase in sent for phrase in remove_phrases):
                continue
            filtered_sentences.append(sent)
        
        # 문장 재결합
        s = ' '.join(filtered_sentences)
        
        return s

def preprocess(json_path, new_json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_contents = []

    for idx, content in enumerate(data):
        # print(content)

        tmp = {}
        title = unicodedata.normalize("NFKC", content['title'].lower())

        # print(f"{idx} 번 데이터 제목: {content.title[:20]}")

        tmp['title'] = clean_doc(title, kin=True, is_title=True)
        tmp['content'] = clean_doc(content['content'], kin=True)
        tmp['link'] = content['link']
        tmp['keyword'] = content['keyword']
        
        if content.content == '':
            continue
        # tmp['idx'] = idx
        new_contents.append(tmp)
    
    with open(new_json_path, "w", encoding='utf-8') as f:
        json.dump(new_contents, f, ensure_ascii=False, indent=4)

if __name__=="__main__":
    keyword_path = 'keywords.txt'

    crawl_data_to_json(keyword_path, 'keyword1_blog.json')
    preprocess('keyword1_blog.json', 'cleaned_data.json')


