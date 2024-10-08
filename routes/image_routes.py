from flask import Blueprint, request, jsonify
from services.image_service import process_image_file
import os

image_blueprint = Blueprint('image', __name__)

@image_blueprint.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    file_name = request.form.get('fileName')

    try:
        response = process_image_file(file, file_name)
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': 'Image processing failed', 'message': str(e)}), 500
