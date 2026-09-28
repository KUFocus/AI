from flask import Blueprint, current_app, jsonify, request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from meeting_answers import MeetingQuestionAnswerer


class MeetingQuestionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    projectId: int = Field(gt=0, le=2**63 - 1)
    minutesId: int | None = Field(default=None, gt=0, le=2**63 - 1)
    question: str = Field(min_length=1, max_length=2000)

    @field_validator('question')
    @classmethod
    def validate_question(cls, value):
        if not value.strip():
            raise ValueError('질문은 공백만으로 작성할 수 없습니다.')
        return value


def create_meeting_answer_blueprint(index_provider, complete):
    """Spring에서 프로젝트 접근 권한을 확인한 뒤 호출하는 내부 질문 API다."""
    blueprint = Blueprint('meeting_answer', __name__)

    @blueprint.route('/answer_meeting', methods=['POST'])
    def answer_meeting():
        try:
            data = MeetingQuestionRequest.model_validate(request.get_json(silent=True))
        except ValidationError:
            return jsonify({'error': '양의 정수 projectId와 공백이 아닌 2000자 이하의 question을 입력해 주세요. minutesId는 선택 사항입니다.'}), 400
        try:
            index = index_provider()
            if index is None:
                return jsonify({'error': '회의록 검색을 위한 로컬 색인 설정이 준비되지 않았습니다.'}), 503
            answerer = MeetingQuestionAnswerer(index, complete)
            result = answerer.answer(data.projectId, data.question, minutes_id=data.minutesId)
            return jsonify(result), 200
        except Exception:
            current_app.logger.exception('회의록 질문 처리 중 오류가 발생했습니다.')
            return jsonify({'error': '회의록 질문을 처리하지 못했습니다.'}), 500

    return blueprint
