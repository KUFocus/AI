import requests
import json
from config.clova_config import CLOVA_SPEECH_INVOKE_URL, CLOVA_SPEECH_API_KEY

def process_audio_file(file_path):
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
