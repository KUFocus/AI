import re
from datetime import date, timedelta

from langchain_core.tools import tool


class ScheduleDateExpressionError(ValueError):
    """원문의 날짜 표현을 확정된 날짜로 해석할 수 없다."""


@tool
def resolve_schedule_date(expression: str, meeting_date: str) -> str:
    """원문의 명시적인 날짜 또는 상대 날짜를 YYYY-MM-DD로 반환한다.

    meeting_date는 한국 기준 회의 날짜를 YYYY-MM-DD로 전달한다.
    오늘, 내일, 모레, 다음 주 요일, YYYY년 M월 D일, YYYY-MM-DD를 지원한다.
    오전 10시, 오후 3시 30분 같은 시각이 붙어도 날짜 부분을 계산한다.
    주의 시작은 월요일이다. 추출 결과의 시간이나 일정의 확정 여부는 판단하지 않는다.
    지원하지 않는 표현은 추측하지 않고 오류로 반환한다.
    """
    try:
        reference_date = date.fromisoformat(meeting_date)
        if reference_date.isoformat() != meeting_date:
            raise ValueError('회의 기준일 형식이 올바르지 않습니다.')
    except ValueError as error:
        raise ValueError('회의 기준일은 YYYY-MM-DD 형식의 유효한 날짜여야 합니다.') from error

    normalized = ''.join(expression.split())
    match = re.fullmatch(
        r'(?P<day>오늘|내일|모레|다음주[월화수목금토일]요일|'
        r'[0-9]{4}년[0-9]{1,2}월[0-9]{1,2}일|[0-9]{4}-[0-9]{2}-[0-9]{2})'
        r'(?:(?P<period>오전|오후)?(?P<hour>[0-9]{1,2})시'
        r'(?:(?P<minute>[0-9]{1,2})분|반)?)?', normalized,
    )
    if not match:
        raise ScheduleDateExpressionError('현재 지원하지 않는 날짜 표현입니다.')
    if match['hour'] is not None:
        hour, minute = int(match['hour']), int(match['minute'] or 0)
        valid_hour = 1 <= hour <= 12 if match['period'] else 0 <= hour <= 23
        if not valid_hour or not 0 <= minute <= 59:
            raise ScheduleDateExpressionError('원문 날짜 표현에 유효하지 않은 시각이 포함되어 있습니다.')

    day = match['day']
    absolute = re.fullmatch(r'([0-9]{4})년([0-9]{1,2})월([0-9]{1,2})일', day)
    if absolute or re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', day):
        try:
            return date(*map(int, absolute.groups())).isoformat() if absolute else date.fromisoformat(day).isoformat()
        except ValueError as error:
            raise ScheduleDateExpressionError('원문 날짜 표현에 유효하지 않은 날짜가 포함되어 있습니다.') from error

    offsets = {'오늘': 0, '내일': 1, '모레': 2}
    if day in offsets:
        days = offsets[day]
    else:
        weekday = '월화수목금토일'.index(day[3])
        days = 7 - reference_date.weekday() + weekday

    try:
        return (reference_date + timedelta(days=days)).isoformat()
    except OverflowError as error:
        raise ValueError('계산한 날짜가 지원 범위를 벗어났습니다.') from error
