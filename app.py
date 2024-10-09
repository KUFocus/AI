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
    app.logger.info(f"Checkpoint loaded successfully. Keys: {checkpoint.keys()}")
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
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "다음 회의록 내용을 바탕으로 핵심을 간결하게 요약하고 마치 제목처럼 만들어 줘. 회의 주제와 목적을 명확하게 나타내는 한 문장으로 간결하게 정리해줘."},
                {"role": "user", "content": input_text}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        summary = response.choices[0].message.content.strip()
        app.logger.info(f"Summarized text: {summary}")
        
        return jsonify({
            'summarizedText': summary,  # Flask 응답을 DTO 필드에 맞춤
            'extractedSchedule': None   # 일정 정보는 없으므로 임시로 None 설정
        }), 200
    except Exception as e:
        app.logger.error(f"Error in summarizing text: {str(e)}")
        return jsonify({'error': 'Summarization failed'}), 500

    
# 음성 파일을 처리하는 엔드포인트
@app.route('/process_audio', methods=['POST'])
def process_audio():
    app.logger.info("Received request for process_audio")
    
    if 'file' not in request.files:
        app.logger.error("No file part in the request")
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    file_name = request.form.get('fileName', 'audio_file')
    
    app.logger.info(f"Received file: {file_name}")

    file_path = os.path.join('/tmp', file_name)
    file.save(file_path)
    app.logger.info(f"File saved at {file_path}")

    try:
        # Clova API 호출
        response = clova_speech_recognition(file_path)
        
        # 응답 데이터에서 필요한 정보만 추출하여 content에 저장할 데이터 구성
        segments = response.get('segments', [])
        content_data = {
            "overall_text": response.get("text", ""),
            "overall_confidence": response.get("confidence", 0.0),
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
    except Exception as e:
        app.logger.error(f"Error in audio processing: {str(e)}")
        return jsonify({'error': 'Audio processing failed'}), 500

# 이미지 파일을 처리하는 엔드포인트 (EasyOCR 사용)
@app.route('/process_image', methods=['POST'])
def process_image():
    app.logger.info("Received request for process_image")

    if 'file' not in request.files:
        app.logger.error("No file part in the request")
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    file_name = request.form.get('fileName')
    app.logger.info(f"Received file: {file_name}")

    try:
        # 이미지 파일 로드
        image = Image.open(file.stream)
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

        response = {
            'filename': file_name,
            'text': corrected_text  # 오탈자 수정된 최종 텍스트만 반환
        }

        # JSON 응답 반환
        return app.response_class(
            response=json.dumps(response, ensure_ascii=False),
            mimetype='application/json'
        )

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
