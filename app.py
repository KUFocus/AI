# -*- coding: utf-8 -*-

import base64
from flask import Flask, request, jsonify
import requests
import json
import os
import logging
import torch
from PIL import Image, UnidentifiedImageError
from io import BytesIO  # Base64 처리용
from torchvision import transforms
import yaml
from custom import Model  # custom.py에서 모델 불러오기
from easyocr import Reader  # EasyOCR 불러오기
import numpy as np
import openai
from dotenv import load_dotenv
from datetime import datetime
from tempfile import NamedTemporaryFile

# .env 파일에서 환경 변수 로드
load_dotenv()

app = Flask(__name__)

# 로그 레벨을 DEBUG로 설정
logging.basicConfig(level=logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.DEBUG)

# Clova Speech API 설정
CLOVA_SPEECH_INVOKE_URL = os.getenv("CLOVA_SPEECH_INVOKE_URL")
CLOVA_SPEECH_API_KEY = os.getenv("CLOVA_SPEECH_API_KEY")

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# 커스텀 모델 로드 설정
model_path = 'custom.pth'
yaml_path = 'custom.yaml'

# YAML 설정 파일 로드
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)

# EasyOCR Reader 초기화
reader = Reader(['ko'], gpu=torch.cuda.is_available())  # EasyOCR은 GPU 사용 가능 여부를 자동 감지

# Model 클래스 인스턴스 생성
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
custom_model = Model(config['network_params']['input_channel'],
                     config['network_params']['output_channel'],
                     config['network_params']['hidden_size'],
                     len(config['character_list']),  # num_class는 character_list의 길이로 설정
                     config['network_params']['num_fiducial'],
                     config['imgH'], config['imgW'],
                     config['network_params']['batch_max_length'])

# 모델 로드 확인 (디버그용)
try:
    checkpoint = torch.load(model_path, map_location=device)
    custom_model.load_state_dict(checkpoint, strict=False)
    custom_model.eval()
    app.logger.info("Model loaded and set to evaluation mode.")
except Exception as e:
    app.logger.error(f"Error loading model: {str(e)}")
    raise e

# 이미지 전처리 설정 (config.yaml 기반)
transform = transforms.Compose([
    transforms.Resize((config['imgH'], config['imgW'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.route('/summarize_text', methods=['POST'])
def summarize_text():
    app.logger.info("Received request for text summarization")
    
    data = request.json
    input_text = data.get('text', '')
    
    if not input_text:
        app.logger.error("No text provided for summarization")
        return jsonify({'error': 'No text provided'}), 400

    try:
        # 현재 날짜를 YYYY-MM-DD 형식으로 가져옴
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"다음 회의록 내용을 바탕으로 JSON 객체를 만들기 위해 두 가지 작업을 수행해줘. "
                        f"첫째, 회의 내용에서 핵심을 요약하여 'summarizedText'로 반환해줘. 요약에 아래 일정 내용에 적을 일정과 관련된 내용은 절대 포함시키지 마. "
                        f"둘째, 일정이 포함되어 있다면, 각 일정을 'schedules' 리스트로 반환해줘. 일정이 여러 번 언급되더라도, 최종적으로 확정된 일정만 하나씩 반환해줘. "
                        f"각 일정은 'extractedScheduleDate' (LocalDateTime 형식, 예: 2024-11-06T12:49:15), "
                        f"'extractedScheduleContent'로 JSON 객체를 만들어 줘. "
                        f"일정 내용은 '제출', '완성'과 같이 일정표에 적는 것처럼 만들어줘 일정 내용에는 날짜 정보를 절대 포함하시키지 마. "
                        f"일정이 '오늘', '내일', '다음 주', '다음주 목요일'과 같은 상대적인 표현일 경우, 오늘의 날짜({today_date})를 기준으로 해당 날짜를 올바른 LocalDateTime 형식으로 환산해줘. "
                        f"만약 일정이 없다면 빈 리스트로 반환해."
                    )
                },
                {"role": "user", "content": input_text}
            ],
            max_tokens=500,
            temperature=0.7
        )
        response_content = response.choices[0].message.content.strip()
        
        app.logger.info(f"GPT response: {response_content}")
        
        # 응답에서 코드 블록을 제거
        if response_content.startswith("```json") and response_content.endswith("```"):
            response_content = response_content[7:-3].strip()

        extracted_data = json.loads(response_content)
        
        return jsonify({
            'summarizedText': extracted_data.get('summarizedText', ''),
            'schedules': extracted_data.get('schedules', [])
        }), 200
    except Exception as e:
        app.logger.error(f"Error in summarizing text: {str(e)}")
        return jsonify({'error': 'Summarization failed'}), 500

    
# 음성 파일을 처리하는 엔드포인트
@app.route('/process_audio', methods=['POST'])
def process_audio():
    app.logger.info("Received request for process_audio")

    data = request.get_json()
    file_url = data.get('filePath', None)

    if not file_url:
        app.logger.error("No filePath provided in the request")
        return jsonify({'error': 'No filePath provided'}), 400

    app.logger.info(f"Received file path: {file_url}")

    try:
        # URL에서 오디오 파일을 다운로드하여 임시 파일에 저장
        with NamedTemporaryFile(delete=True, suffix=".mp3") as temp_file:
            response = requests.get(file_url, stream=True)
            if response.status_code == 200:
                temp_file.write(response.content)
                temp_file.flush()

                # Clova API 호출
                clova_response = clova_speech_recognition(temp_file.name)
                
                # 응답 데이터에서 필요한 정보만 추출하여 content에 저장할 데이터 구성
                segments = clova_response.get('segments', [])
                content_data = {
                    "overall_text": clova_response.get("text", ""),
                    "overall_confidence": clova_response.get("confidence", 0.0),
                    "segments": [
                        {
                            "speaker": segment["speaker"]["name"],
                            "text": segment["text"],
                            "confidence": round(segment["confidence"], 4),
                            "start_time": segment["start"],
                            "end_time": segment["end"]
                        }
                        for segment in segments
                    ]
                }
                
                # JSON 형식으로 content에 저장
                content = json.dumps(content_data, ensure_ascii=False)

                return app.response_class(
                    response=content,
                    mimetype='application/json'
                )
            else:
                app.logger.error("Failed to download file from URL")
                return jsonify({'error': 'Failed to download file from URL'}), 404

    except Exception as e:
        app.logger.error(f"Audio processing error: {str(e)}")
        return jsonify({'error': 'Audio processing failed'}), 500

# 이미지 파일을 처리하는 엔드포인트 (EasyOCR 사용)
@app.route('/process_image', methods=['POST'])
def process_image():
    app.logger.info("Received request for process_image")

    data = request.get_json()
    file_url = data.get('filePath', None)

    if not file_url:
        app.logger.error("No filePath provided in the request")
        return jsonify({'error': 'No filePath provided'}), 400

    app.logger.info(f"Received file path: {file_url}")

    try:
        # URL에서 이미지를 다운로드하여 임시 파일에 저장
        with NamedTemporaryFile(delete=True, suffix=".jpg") as temp_file:
            response = requests.get(file_url, stream=True)
            if response.status_code == 200:
                temp_file.write(response.content)
                temp_file.flush()
                
                # 다운로드된 이미지를 로드
                with Image.open(temp_file.name) as image:
                    app.logger.info(f"Image format: {image.format}")

                    # 이미지 크기 확인 후 리사이즈
                    max_size = (1024, 1024)
                    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                        app.logger.info(f"Image is too large, resizing to {max_size}")
                        image.thumbnail(max_size)

                    app.logger.info(f"Resized image size: {image.size}")

                    # EasyOCR로 텍스트 감지 및 인식
                    results = reader.readtext(np.array(image), detail=0)
                    detected_text = ' '.join(results)
                    app.logger.info(f"Detected text: {detected_text}")

                    # GPT를 이용한 오탈자 수정
                    corrected_text = correct_spelling_with_gpt(detected_text)

                    response_data = {
                        'filename': os.path.basename(file_url),
                        'text': corrected_text
                    }

                    return app.response_class(
                        response=json.dumps(response_data, ensure_ascii=False),
                        mimetype='application/json'
                    )
            else:
                app.logger.error("Failed to download file from URL")
                return jsonify({'error': 'Failed to download file from URL'}), 404

    except UnidentifiedImageError as e:
        app.logger.error(f"Image processing error: {str(e)}")
        return jsonify({'error': 'Invalid image file'}), 400
    except Exception as e:
        app.logger.error(f"Image processing error: {str(e)}")
        return jsonify({'error': 'Image processing failed'}), 500


def correct_spelling_with_gpt(detected_text):
    """ GPT를 이용하여 오탈자를 교정하는 함수 """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "다음 텍스트의 오탈자를 교정해 줘. 교정된 텍스트를 제외하고 어떤 문자도 적지마."},
                {"role": "user", "content": detected_text}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        corrected_text = response.choices[0].message.content.strip()
        app.logger.info(f"Corrected text: {corrected_text}")
        return corrected_text
    except Exception as e:
        app.logger.error(f"Error in correcting text with GPT: {str(e)}")
        return detected_text  # 교정 실패 시 원본 텍스트 반환


# Clova Speech API를 호출하여 파일을 텍스트로 변환하는 함수
def clova_speech_recognition(file_path):
    request_body = {
        'language': 'ko-KR',
        'completion': 'sync',
        'wordAlignment': True,
        'fullText': True,
        'diarization': {'enable': True},
        'format': 'JSON'
    }

    headers = {
        'X-CLOVASPEECH-API-KEY': CLOVA_SPEECH_API_KEY
    }

    files = {
        'media': open(file_path, 'rb'),
        'params': (None, json.dumps(request_body), 'application/json')
    }

    response = requests.post(f"{CLOVA_SPEECH_INVOKE_URL}/recognizer/upload", headers=headers, files=files)
    return response.json()

# 모델 출력 해석하여 텍스트로 변환하는 함수
def decode_model_output(output):
    app.logger.info(f"Decoding model output with shape: {output.shape}")
    output = output.permute(1, 0, 2)  # (T, N, C)로 변환
    app.logger.info(f"Permuted output shape: {output.shape}")
    
    _, max_indices = output.max(2)
    app.logger.info(f"Max indices: {max_indices}")
    
    max_indices = max_indices.view(-1)

    decoded_text = []
    prev_idx = -1
    for idx in max_indices:
        if idx != prev_idx:
            if idx != 0:  # 0은 공백 문자로 간주
                decoded_text.append(config['character_list'][idx])
        prev_idx = idx

    return ''.join(decoded_text)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
