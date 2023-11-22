import streamlit as st
import os
import base64
import requests
from PIL import Image
from os.path import isdir
#SMTP
from email.message import EmailMessage
from smtplib import SMTP_SSL
from pathlib import Path
from haisecret import MY_ID, MY_PW
#전처리
from sheet2png import SHEETtoPNG
from matplotlib import pyplot as plt

def main():
    st.title("Handwriting Font Generation")
    st.header("손글씨 폰트 생성")
    st.markdown("**HAI 2023 project 3팀**")
    st.subheader("")

    # 이미지가 제공되는 웹 URL 또는 로컬 파일 시스템 경로
    image_url_or_path = "./handwrite_sample.png"

    # 이미지를 다운로드할 수 있는 링크 생성
    download_link = create_download_link_from_url_or_path(image_url_or_path)

    # "손글씨 작성 폼 다운로드" 문구 표시
    st.subheader("손글씨 작성 폼 다운로드")
    download = st.button("다운로드")
    if download: st.markdown(download_link, unsafe_allow_html=True)
    st.subheader("")

    # "손글씨 폼 다운로드" 문구 표시
    st.subheader("손글씨 이미지 등록")

    # 사용자에게 이미지 파일을 업로드하도록 요청
    uploaded_file = st.file_uploader("이미지 파일을 업로드하세요.", type=["jpg", "jpeg", "png"])
    
    # 사용자에게 완성된 폰트를 전송 받을 이메일 주소를 입력하도록 요청
    email_address = st.text_input("폰트를 전송 받을 email 주소를 입력하세요.", value=None)
    agree = False
    if not email_address:
        st.warning("email 주소를 입력해주세요.")
    else:
        agree = st.checkbox("개인 정보 제공에 동의합니다.")
    
    if agree:
        save_path = ""
        if uploaded_file is not None:
            # 업로드된 이미지를 특정 폴더에 저장            
            save_path = save_uploaded_image(uploaded_file, email_address)
            st.success(f"이미지가 성공적으로 저장되었습니다.")

        # 이미지 전처리 진행
        # 또한 전처리된 이미지를 사이트에 표시
        sheet_to_png(save_path, email_address) # save_path는 form 저장 위치
    
        #####################################
        # Git에서 모델 불러와서 png파일 생성  #
        # 그 후 여러가지 후처리              #
        #####################################
        ttf_file = False
        ttf_file = "test.png" # ttf 파일로 수정해야함

        # 사용자에게 완성된 폰트를 이메일로 전송
        if(ttf_file):
            send_mail(email_address, "나만의 손글씨 TTF 파일", "세상에 하나뿐인 나만의 손글씨 TTF 파일을 선물합니다!", ttf_file)
            st.success(f"손글씨 파일이 성공적으로 전송되었습니다.")

def create_download_link_from_url_or_path(image_url_or_path):
    """
    웹 URL 또는 로컬 파일 시스템 경로에서 이미지를 다운로드할 수 있는 HTML 링크 생성
    """
    # 이미지를 바이트로 변환
    if image_url_or_path.startswith("http"):
        image_bytes = requests.get(image_url_or_path).content
    else:
        with open(image_url_or_path, "rb") as image_file:
            image_bytes = image_file.read()

    # 이미지를 base64로 인코딩
    encoded_image = base64.b64encode(image_bytes).decode()

    # 다운로드할 수 있는 링크 생성
    href = f'<a href="data:image/jpeg;base64,{encoded_image}" download="downloaded_image.jpg">Click here to download</a>'
    return href

def save_uploaded_image(uploaded_file, email_address):
    """
    업로드된 이미지를 특정 폴더에 저장하고 저장된 경로를 반환
    """
    # 특정 폴더를 만들거나 이미 존재하면 그대로 사용
    target_folder = email_address + "/uploaded_images"
    os.makedirs(target_folder, exist_ok=True)

    # 업로드된 이미지를 특정 폴더에 저장                            
    save_path = os.path.join(target_folder, uploaded_file.name) 
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    return save_path

def show_registered_images(registered_images):
    """
    현재 등록된 이미지 리스트를 표시
    """
    if not registered_images:
        st.write("현재 등록된 이미지가 없습니다.")
    else:
        st.write("현재 등록된 이미지 리스트:")
        for image_name in registered_images:
            st.write(image_name)

# 이미지 전처리가 진행
# 또한 전처리된 이미지를 사이트에 표시
def sheet_to_png(save_path, email_address): # save_path는 form 저장 위치
        base_folder_path = "./"+email_address+"/pretreatment"
        convertor = SHEETtoPNG()
        convertor.convert(save_path, base_folder_path)

        image_per_column = 10
        colums = st.columns(image_per_column)
        for folder_num in range(33, 123):
            folder_path = os.path.join(base_folder_path, str(folder_num))
            if(isdir(folder_path)):
                image_file = get_image_file(folder_path)
                image = Image.open(image_file)

                current_column = colums[(folder_num-33) % image_per_column]
                with current_column:
                    st.image(image, caption=f"Image {folder_num}", width=70)
            else:
                current_column = colums[(folder_num-33) % image_per_column]
                with current_column:
                    st.image(Image.open("./nullimage.png"), caption=f"Null Image", width=70)

# 전처리된 이미지 불러오기
def get_image_file(folder_path):
    image_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith((".jpg", ".jpeg", ".png"))]
    return image_files[0] if image_files else None

#이메일 전송
def send_mail(receiver, title, text, font=False):
    # 템플릿 생성
    msg = EmailMessage()

    # 보내는 사람 / 받는 사람 / 제목 입력
    msg["From"] = MY_ID
    msg["To"] = receiver
    msg["Subject"] = title

    # 본문 구성
    msg.set_content(text)
    
    # 파일 첨부
    if font:
        파일명 = Path(font).name
        with open(font, "rb") as f:
            msg.add_attachment(f.read(), maintype="application", subtype="octet-stream", filename=파일명)
            msg.add_header('Content-Disposition', 'attachment', filename=파일명)

    with SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(MY_ID, MY_PW)
        smtp.send_message(msg)
    
    # 완료 메시지
    print(receiver, "성공", sep="\t")

if __name__ == "__main__":
    main()
