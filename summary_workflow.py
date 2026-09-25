import json
import logging
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from schedule_history import ScheduleHistoryError, resolve_schedule_history
from schedule_tools import (
    ScheduleDateExpressionError, ScheduleTimeExpressionError,
    resolve_schedule_date, resolve_schedule_time,
)


logger = logging.getLogger(__name__)


class SummaryState(TypedDict, total=False):
    messages: list[dict[str, str]]
    input_text: str
    meeting_date: str
    request_datetime: str
    omitted_schedule_indexes: list[int]
    response_content: str
    result: dict | None
    attempts: int
    validation_error: str | None
    date_checks: list[dict]
    time_checks: list[dict]
    schedule_indexes: list[int]


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

    def resolve_histories(state: SummaryState):
        groups = {}
        for index, decision in enumerate(state['result']['schedules']):
            groups.setdefault(decision['eventId'], []).append((index, decision))
        schedules = []
        indexes = []
        omitted = []
        for event_id, entries in groups.items():
            decisions = [{key: value for key, value in decision.items() if key != 'eventId'}
                         for _, decision in entries]
            try:
                selected = resolve_schedule_history(decisions, state['input_text'])
            except ScheduleHistoryError as error:
                feedback = f'일정 식별자 {event_id}: {error}'
                if state['attempts'] >= max_attempts:
                    raise ScheduleHistoryError(feedback) from error
                return {'result': None, 'validation_error': feedback, 'schedule_indexes': []}
            if selected is not None:
                source_index = next(index for index, decision in entries
                                    if decision['evidence'] == selected['evidence'])
                if selected['dateExpression'] is None and selected['timeExpression'] is None:
                    omitted.append(source_index)
                    logger.info('일정 %s는 날짜와 시각 근거가 모두 없어 저장 대상에서 제외했습니다.', source_index)
                    continue
                schedules.append(selected)
                indexes.append(source_index)
        return {
            'result': {**state['result'], 'schedules': schedules},
            'schedule_indexes': indexes, 'omitted_schedule_indexes': omitted, 'validation_error': None,
        }

    def normalize_dates(state: SummaryState):
        checks = []
        errors = []
        schedules = []
        for index, schedule in zip(state['schedule_indexes'], state['result']['schedules'], strict=True):
            try:
                if schedule['dateExpression'] is None:
                    expected_date = state['meeting_date']
                else:
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

            actual_date, _, clock = (schedule['extractedScheduleDate'] or '').partition('T')
            matches = actual_date == expected_date
            check = {
                'schedule_index': index, 'status': 'matched' if matches else 'normalized',
                'expected_date': expected_date,
            }
            if not matches:
                check['original_date'] = actual_date
                logger.info(
                    '일정 %s의 날짜를 원문과 회의 기준일에 따라 보정했습니다. %s -> %s',
                    index, actual_date, expected_date,
                )
            if schedule['dateExpression'] is None:
                check.update(status='defaulted', policy='reference_date')
                logger.info('일정 %s는 날짜 표현이 없어 기준일 %s를 적용했습니다.', index, expected_date)
            checks.append(check)
            normalized_date = f'{expected_date}T{clock}' if clock else expected_date
            schedules.append({**schedule, 'extractedScheduleDate': normalized_date})
        if errors:
            feedback = '\n'.join(errors)
            if state['attempts'] >= max_attempts:
                raise ValueError(feedback)
            return {'result': None, 'validation_error': feedback, 'date_checks': checks}
        return {
            'result': {**state['result'], 'schedules': schedules},
            'validation_error': None, 'date_checks': checks,
        }

    def normalize_times(state: SummaryState):
        checks = []
        errors = []
        schedules = []
        for index, schedule in zip(state['schedule_indexes'], state['result']['schedules'], strict=True):
            expression = schedule['timeExpression']
            try:
                clock = '18:00:00' if expression is None else resolve_schedule_time.invoke({
                    'expression': expression, 'assume_business_hours': True,
                })
            except ScheduleTimeExpressionError as error:
                checks.append({'schedule_index': index, 'status': 'unresolved'})
                errors.append(f'schedules.{index}.timeExpression: {error} 원문의 시각 근거와 조건을 유지해 주세요.')
                continue
            day, _, original_clock = schedule['extractedScheduleDate'].partition('T')
            matches = original_clock == clock
            check = {'schedule_index': index, 'status': 'matched' if matches else 'normalized', 'expected_time': clock}
            if not matches:
                check['original_time'] = original_clock
                logger.info('일정 %s의 시각을 계산 결과로 보정했습니다. %s -> %s', index, original_clock, clock)
            if expression is None:
                check.update(status='defaulted', policy='missing_time_18')
                logger.info('일정 %s는 시각 표현이 없어 기본 시각 18:00을 적용했습니다.', index)
            elif re.match(r'^(?:0?[1-9]|1[0-2])\s*시', expression.strip()):
                check.update(status='defaulted', policy='business_hours')
                logger.info('일정 %s는 오전과 오후가 없어 업무시간 규칙으로 %s를 적용했습니다.', index, clock)
            timestamp = f'{day}T{clock}'
            if state.get('request_datetime'):
                requested_at = datetime.fromisoformat(state['request_datetime'])
                scheduled_at = datetime.fromisoformat(timestamp).replace(tzinfo=ZoneInfo('Asia/Seoul'))
                check['before_request'] = scheduled_at < requested_at
                if check['before_request']:
                    logger.info('일정 %s는 최초 처리 시각보다 이전입니다. 날짜를 자동으로 미루지 않습니다.', index)
            checks.append(check)
            schedules.append({**schedule, 'extractedScheduleDate': timestamp})
        if errors:
            feedback = '\n'.join(errors)
            if state['attempts'] >= max_attempts:
                raise ValueError(feedback)
            return {'result': None, 'validation_error': feedback, 'time_checks': checks}
        return {
            'result': {**state['result'], 'schedules': schedules},
            'validation_error': None, 'time_checks': checks,
        }

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
    graph.add_node('resolve_histories', resolve_histories)
    graph.add_node('normalize_dates', normalize_dates)
    graph.add_node('normalize_times', normalize_times)
    graph.add_node('repair', repair)
    graph.add_edge(START, 'generate')
    graph.add_edge('generate', 'validate')
    graph.add_conditional_edges('validate', route_after_validation, {'repair': 'repair', 'end': 'resolve_histories'})
    graph.add_conditional_edges('resolve_histories', route_after_validation, {'repair': 'repair', 'end': 'normalize_dates'})
    graph.add_conditional_edges('normalize_dates', route_after_validation, {'repair': 'repair', 'end': 'normalize_times'})
    graph.add_conditional_edges('normalize_times', route_after_validation, {'repair': 'repair', 'end': END})
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
