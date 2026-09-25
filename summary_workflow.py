import json
import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError


logger = logging.getLogger(__name__)


class SummaryState(TypedDict, total=False):
    messages: list[dict[str, str]]
    input_text: str
    response_content: str
    result: dict | None
    attempts: int
    validation_error: str | None


def build_summary_workflow(generate_response, validate_response, *, repair_invalid_response=True):
    max_attempts = 2 if repair_invalid_response else 1

    def generate(state: SummaryState):
        return {
            'response_content': generate_response(state['messages']),
            'attempts': state.get('attempts', 0) + 1,
        }

    def validate(state: SummaryState):
        try:
            result = validate_response(state['response_content'], state['input_text'])
            return {'result': result, 'validation_error': None}
        except (json.JSONDecodeError, ValidationError) as error:
            if state['attempts'] >= max_attempts:
                raise
            return {'result': None, 'validation_error': validation_feedback(error)}

    def route_after_validation(state: SummaryState):
        return 'repair' if state['validation_error'] else 'end'

    def repair(state: SummaryState):
        logger.warning('모델 응답 검증에 실패하여 한 번 수정을 요청합니다. %s', state['validation_error'])
        return {'messages': state['messages'] + [
            {'role': 'assistant', 'content': state['response_content']},
            {'role': 'user', 'content': (
                f"검증 오류: {state['validation_error']}\n"
                '원래 회의 내용과 기준일을 유지하고 오류를 수정한 전체 JSON 응답을 작성해 주세요. '
                '필수 항목과 자료형, 실제 날짜와 시간, 비어 있지 않은 일정 내용을 확인해 주세요. '
                '검증을 통과하려고 원문에 없는 일정을 만들거나 일정을 임의로 삭제하지 마세요.'
            )},
        ]}

    graph = StateGraph(SummaryState)
    graph.add_node('generate', generate)
    graph.add_node('validate', validate)
    graph.add_node('repair', repair)
    graph.add_edge(START, 'generate')
    graph.add_edge('generate', 'validate')
    graph.add_conditional_edges('validate', route_after_validation, {'repair': 'repair', 'end': END})
    graph.add_edge('repair', 'generate')
    return graph.compile()


def validation_feedback(error: json.JSONDecodeError | ValidationError) -> str:
    if isinstance(error, json.JSONDecodeError):
        return '응답이 올바른 JSON 형식이 아닙니다.'
    reasons = {
        'missing': '필수 항목이 누락되었습니다.',
        'extra_forbidden': '정의하지 않은 항목입니다.',
        'string_type': '문자열이어야 합니다.',
        'list_type': '목록이어야 합니다.',
        'model_type': '객체 형태여야 합니다.',
    }
    feedback = []
    for item in error.errors(include_input=False, include_url=False):
        field = '.'.join(map(str, item['loc'])) or '응답 전체'
        reason = str(item['ctx']['error']) if item['type'] == 'value_error' else reasons.get(
            item['type'], '값이 검증 조건을 충족하지 않습니다.'
        )
        feedback.append(f'{field}: {reason}')
    return '\n'.join(feedback)
