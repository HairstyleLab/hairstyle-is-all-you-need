import os
import uuid
import datetime
import streamlit as st
import base64
from test import encode_image_from_file, make_human_message

# =========================
# 기본 설정 & 유틸
# =========================
st.set_page_config(
    page_title="헤어스타일 상담 챗봇 데모",
    layout="wide"
)

UPLOAD_DIR = "uploaded_images"
GENERATED_DIR = "generated_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # {role, content, images}


def save_uploaded_files(uploaded_files):
    """업로드된 파일들을 디스크에 저장하고 경로 리스트 반환"""
    if not uploaded_files:
        return None

    saved_paths = []
    for file in uploaded_files:
        ext = os.path.splitext(file.name)[1]
        unique_name = (
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S") +
            "_" + uuid.uuid4().hex + ext
        )
        save_path = os.path.join(UPLOAD_DIR, unique_name)
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(save_path)

    return saved_paths


def call_model(user_text, user_image_paths):
    """여기가 실제 모델 연동 포인트 (지금은 데모용 더미 응답)"""
    assistant_text = (
        "여기는 헤어스타일 상담 챗봇의 데모 응답입니다.\n\n"
        f"- 사용자가 보낸 텍스트: `{user_text}`\n"
        f"- 함께 온 이미지 개수: {len(user_image_paths)}장\n\n"
        "이 자리에 실제 모델의 답변 텍스트를 넣으면 됩니다."
    )

    generated_image_paths = []
    # 실제론 여기서 생성 이미지 저장 후 generated_image_paths에 경로 append

    return assistant_text, generated_image_paths

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 사용하려는 로컬 이미지 파일 이름 (같은 폴더에 있어야 편함)
img_file = "images/logo.png"
img_base64 = get_base64_of_bin_file(img_file)

# =========================
# 메인 앱
# =========================
def main():
    init_session_state()
    st.markdown(f"""
    <style>
        header[data-testid="stHeader"] {{
            background: transparent !important;
            border-bottom: none !important; 
            pointer-events: none !important;
        }}
        
        .block-container {{
            padding-top: 5rem !important;
            padding-bottom: 5rem !important;
        }}

        .custom-navbar {{
            position: fixed;
            top: 0px;
            left: 0;
            width: 100%;
            height: 60px;
            background-color: #1E1E1E;
            color: white;
            z-index: 999999;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .nav-logo {{
            font-size: 1.2rem;
            font-weight: bold;
        }}
                
        .logo-link, .logo-link:visited, .logo-link:active {{
            text-decoration: none !important;
            color: white !important;
            display: flex;
            align-items: center;
        }}

        .logo-link:hover {{
            text-decoration: none !important;
            color: #1e1ef0 !important;
        }}
    </style>

    <div class="custom-navbar">
        <div class="nav-logo" style="display: flex; align-items: center;">
            <a href="/" target="_self" class="logo-link">
                <img src="data:image/png;base64,{img_base64}" 
                    style="height: 70px; margin-right: 5px;">
                <span>Hairstyle is all you need</span>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- 채팅 히스토리 ----------
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg.get("content"):
                    st.markdown(msg["content"])
                
                if img_path := msg.get("images"):
                    st.image(img_path, width=220)

                # for img_path in msg.get("images", []):
                #     st.image(img_path, width=220)

    user_text = st.chat_input("메시지를 입력하세요")  # Enter 시 전송
    uploader_key = f"chat_uploader_{len(st.session_state.messages)}"
    with st._bottom:
        uploaded_files = st.file_uploader(
                "이미지 업로드",
                type=["png", "jpg", "jpeg"],
                accept_multiple_files=True,
                key=uploader_key,
            )
    # ---------- 전송 처리 ----------
    if user_text:

        # 1) 업로드된 이미지 저장
        user_image_paths = save_uploaded_files(uploaded_files)

        # 2) USER 메시지 추가
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_text,
                "images": user_image_paths,
            }
        )
        
        with st.chat_message("user"):
            st.markdown(user_text)
        
        with st.chat_message("assistant"):
            with st.spinner("응답 생성중..."):
                # 3) 모델 호출
                # assistant_text, assistant_image_paths = call_model(user_text, user_image_paths)
                # if user_image_paths:
                #     encoded_image = encode_image_from_file(user_image_paths[0])
                assistant_response, flag = make_human_message(user_text, 'test1', user_image_paths)
                # import time
                # time.sleep(2)

            folder_path = "./results"
            path = len([file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]) - 1

            # 4) ASSISTANT 메시지 추가
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": assistant_response['output'],
                    # "content": "test content",
                    "images": f'results/{path}.jpg' if flag else None,
                }
            )

            st.rerun()


if __name__ == "__main__":
    main()
