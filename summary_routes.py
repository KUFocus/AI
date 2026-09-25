from datetime import date

from flask import Blueprint, current_app, jsonify, request


def create_summary_blueprint(summarize):
    blueprint = Blueprint('summary', __name__)

    @blueprint.route('/summarize_text', methods=['POST'])
    def summarize_text():
        current_app.logger.info("Received request for text summarization")

        data = request.json
        input_text = data.get('text', '')

        if not input_text:
            current_app.logger.error("요약할 회의 내용이 없습니다.")
            return jsonify({'error': '요약할 회의 내용을 입력해 주세요.'}), 400

        meeting_date = None
        raw_meeting_date = data.get('meetingDate')
        if raw_meeting_date is not None:
            try:
                meeting_date = date.fromisoformat(raw_meeting_date)
                if meeting_date.isoformat() != raw_meeting_date:
                    raise ValueError('날짜 형식은 YYYY-MM-DD여야 합니다.')
            except (TypeError, ValueError):
                return jsonify({'error': '회의 날짜(meetingDate)는 YYYY-MM-DD 형식의 유효한 날짜여야 합니다.'}), 400

        try:
            return jsonify(summarize(input_text, meeting_date=meeting_date)), 200
        except Exception:
            current_app.logger.exception("회의 내용 요약 중 오류가 발생했습니다.")
            return jsonify({'error': '회의 내용을 요약하지 못했습니다.'}), 500

    return blueprint
