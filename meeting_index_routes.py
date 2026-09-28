from meeting_index import LocalMeetingIndex, DeletedMeetingError

from flask import Blueprint, current_app, jsonify, request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class MeetingIndexIdentity(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    projectId: int = Field(gt=0, le=2**63 - 1)
    minutesId: int = Field(gt=0, le=2**63 - 1)


class IndexMeetingRequest(MeetingIndexIdentity):
    text: str = Field(min_length=1, max_length=100000)

    @field_validator('text')
    @classmethod
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError('회의 원문은 공백만으로 이루어질 수 없습니다.')
        return value


def create_meeting_index_blueprint(index_provider, database_path=None):
    """Spring에서 접근 권한을 확인한 뒤 호출할 내부 색인 API다."""
    blueprint = Blueprint('meeting_index', __name__)

    @blueprint.route('/index_meeting', methods=['POST'])
    def index_meeting():
        try:
            data = IndexMeetingRequest.model_validate(request.get_json(silent=True))
        except ValidationError:
            return jsonify({'error': '양의 정수 projectId와 minutesId, 공백이 아닌 100000자 이하의 text를 JSON으로 입력해 주세요.'}), 400
        try:
            index = index_provider()
            if index is None:
                return jsonify({'error': '회의록 색인에 필요한 로컬 모델과 저장 경로 설정이 준비되지 않았습니다.'}), 503
            result = index.index(data.projectId, data.minutesId, data.text)
            current_app.logger.info('회의록 색인을 완료했습니다. 프로젝트 ID=%s, 회의록 ID=%s, 변경 여부=%s',
                                    data.projectId, data.minutesId, result.changed)
            return jsonify({'projectId': data.projectId, 'minutesId': data.minutesId,
                            'changed': result.changed, 'chunkCount': result.chunk_count,
                            'sourceHash': result.source_hash}), 200
        except DeletedMeetingError:
            return jsonify({'error': '삭제된 회의록은 다시 색인할 수 없습니다.'}), 409
        except Exception:
            current_app.logger.exception('회의록 색인 처리 중 오류가 발생했습니다.')
            return jsonify({'error': '회의록을 색인하지 못했습니다.'}), 500

    @blueprint.route('/index_meeting', methods=['DELETE'])
    def delete_meeting_index():
        try:
            data = MeetingIndexIdentity.model_validate(request.get_json(silent=True))
        except ValidationError:
            return jsonify({'error': '양의 정수 projectId와 minutesId를 JSON으로 입력해 주세요.'}), 400
        if not isinstance(database_path, str) or not database_path.strip():
            return jsonify({'error': '회의록 색인 저장 경로가 설정되지 않았습니다.'}), 503
        try:
            deleted = LocalMeetingIndex.delete_saved(database_path, data.projectId, data.minutesId)
            current_app.logger.info('회의록 색인 삭제를 처리했습니다. 프로젝트 ID=%s, 회의록 ID=%s, 삭제 여부=%s',
                                    data.projectId, data.minutesId, deleted)
            return jsonify({'projectId': data.projectId, 'minutesId': data.minutesId, 'deleted': deleted}), 200
        except Exception:
            current_app.logger.exception('회의록 색인 삭제 중 오류가 발생했습니다.')
            return jsonify({'error': '회의록 색인을 삭제하지 못했습니다.'}), 500

    return blueprint
