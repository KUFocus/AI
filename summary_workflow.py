import json
import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from schedule_tools import ScheduleDateExpressionError, resolve_schedule_date


logger = logging.getLogger(__name__)


class SummaryState(TypedDict, total=False):
    messages: list[dict[str, str]]
    input_text: str
    meeting_date: str
    response_content: str
    result: dict | None
    attempts: int
    validation_error: str | None
    date_checks: list[dict]


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

    def validate_dates(state: SummaryState):
        checks = []
        errors = []
        for index, schedule in enumerate(state['result']['schedules']):
            if schedule['status'] != 'confirmed':
                continue
            try:
                expected_date = resolve_schedule_date.invoke({
                    'expression': schedule['dateExpression'],
                    'meeting_date': state['meeting_date'],
                })
            except ScheduleDateExpressionError as error:
                checks.append({'schedule_index': index, 'status': 'unresolved'})
                errors.append(
                    f'schedules.{index}.dateExpression: {error} '
                    '원문의 날짜 근거를 다시 확인해 주세요. '
                    '임의의 날짜로 바꾸거나 불명확한 조건을 삭제하지 마세요.'
                )
                continue

            actual_date = schedule['extractedScheduleDate'].split('T')[0]
            matches = actual_date == expected_date
            checks.append({
                'schedule_index': index, 'status': 'matched' if matches else 'mismatch',
                'expected_date': expected_date,
            })
            if not matches:
                errors.append(
                    f'schedules.{index}.extractedScheduleDate: '
                    f"회의 기준일 {state['meeting_date']}에서 '{schedule['dateExpression']}'을 "
                    f'계산한 날짜는 {expected_date}이지만 응답 날짜는 {actual_date}입니다.'
                )
        if errors:
            feedback = '\n'.join(errors)
            if state['attempts'] >= max_attempts:
                raise ValueError(feedback)
            return {'result': None, 'validation_error': feedback, 'date_checks': checks}
        return {'validation_error': None, 'date_checks': checks}

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
    graph.add_node('validate_dates', validate_dates)
    graph.add_node('repair', repair)
    graph.add_edge(START, 'generate')
    graph.add_edge('generate', 'validate')
    graph.add_conditional_edges('validate', route_after_validation, {'repair': 'repair', 'end': 'validate_dates'})
    graph.add_conditional_edges('validate_dates', route_after_validation, {'repair': 'repair', 'end': END})
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
        'literal_error': '일정 상태는 confirmed, tentative, cancelled 중 하나여야 합니다.',
    }
    feedback = []
    for item in error.errors(include_input=False, include_url=False):
        field = '.'.join(map(str, item['loc'])) or '응답 전체'
        reason = str(item['ctx']['error']) if item['type'] == 'value_error' else reasons.get(
            item['type'], '값이 검증 조건을 충족하지 않습니다.'
        )
        feedback.append(f'{field}: {reason}')
    return '\n'.join(feedback)
