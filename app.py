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
from tempfile import NamedTemporaryFile
from summarization import MeetingSummarizer
from summary_model import OpenAISummaryModel
from summary_routes import create_summary_blueprint
from meeting_index_routes import create_meeting_index_blueprint
from meeting_index_runtime import create_local_index_provider

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

summarizer = MeetingSummarizer(
    OpenAISummaryModel(openai),
    repair_complete=OpenAISummaryModel(openai, max_tokens=1000),
)
app.register_blueprint(create_summary_blueprint(summarizer.summarize))
app.register_blueprint(create_meeting_index_blueprint(create_local_index_provider(
    os.getenv('MEETING_EMBEDDING_MODEL_DIR'),
    os.getenv('MEETING_INDEX_DATABASE'),
    os.getenv('MEETING_EMBEDDING_REVISION'),
)))

    
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

                    # 1단계: EasyOCR로 텍스트 감지 및 인식
                    easyocr_results = reader.readtext(np.array(image), detail=0)
                    easyocr_text = ' '.join(easyocr_results)
                    app.logger.info(f"EasyOCR detected text: {easyocr_text}")

                    # 2단계: GPT-4o OCR로 텍스트 감지 및 인식
                    corrected_text = gpt_ocr_for_image(file_url, easyocr_text)
                    app.logger.info(f"GPT-4o OCR detected text: {corrected_text}")

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


def gpt_ocr_for_image(file_url, easyocr_text):
    """ GPT-4o를 사용하여 이미지를 분석하고 EasyOCR 결과를 보완하는 함수 """
    try:
        prompt = (
            "다음은 이미지에서 추출된 텍스트야. 이 텍스트는 EasyOCR로 추출된 것으로, 오탈자나 숫자가 잘못 인식될 수 있어."
            "이 텍스트를 참고하여 이미지에서 정확한 모든 텍스트를 다시 인식하고, 수정된 완성된 텍스트를 반환해줘."
            "너가 분석한 텍스트를 최우선적으로 고려해서 수정해주고, 인덴트가 있으면 적용해서 반환해줘."
            "너가 분석한 숫자를 수정된 텍스트에 반영해주고, 1과 (를 헷갈리지 마."
            "수정된 텍스트 이외의 그 어떤 문자도 작성하지 말고, 수정된 텍스트만을 반환해줘."
        )
        
        # GPT에게 EasyOCR 결과와 함께 요청
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Image URL: {file_url}\nEasyOCR Text: {easyocr_text}"}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        gpt_corrected_text = response.choices[0].message.content.strip()
        app.logger.info(f"Corrected text from GPT-4o OCR: {gpt_corrected_text}")
        return gpt_corrected_text

    except Exception as e:
        app.logger.error(f"Error in GPT-4o OCR correction: {str(e)}")
        return easyocr_text  # 실패 시 EasyOCR 결과 반환

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
