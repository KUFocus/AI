from flask import Blueprint, request, jsonify
from services.audio_service import process_audio_file
import os

audio_blueprint = Blueprint('audio', __name__)

@audio_blueprint.route('/process_audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    file_name = request.form.get('fileName', 'audio_file')

    file_path = os.path.join('/tmp', file_name)
    file.save(file_path)

    try:
        response = process_audio_file(file_path)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': 'Audio processing failed', 'message': str(e)}), 500
