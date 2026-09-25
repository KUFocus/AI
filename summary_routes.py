from flask import Blueprint, current_app, jsonify, request


def create_summary_blueprint(summarize):
    blueprint = Blueprint('summary', __name__)

    @blueprint.route('/summarize_text', methods=['POST'])
    def summarize_text():
        current_app.logger.info("Received request for text summarization")

        data = request.json
        input_text = data.get('text', '')

        if not input_text:
            current_app.logger.error("No text provided for summarization")
            return jsonify({'error': 'No text provided'}), 400

        try:
            return jsonify(summarize(input_text)), 200
        except Exception:
            current_app.logger.exception("Error in summarizing text")
            return jsonify({'error': 'Summarization failed'}), 500

    return blueprint
