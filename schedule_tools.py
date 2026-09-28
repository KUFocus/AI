import re
from datetime import date, timedelta

from langchain_core.tools import tool


# 한 시점을 저장하는 계약에서 날짜나 시각 뒤에 붙는 조사만 허용한다.
# 근사, 범위, 선택, 부정 표현은 지우지 않고 전체 표현 검증에서 거부한다.
_TEMPORAL_PARTICLE = r'(?:에(?:는|도)?|까지(?:는|도|로)?|부터(?:는|도)?|로(?:는|도)?)?'


class ScheduleDateExpressionError(ValueError):
    """원문의 날짜 표현을 확정된 날짜로 해석할 수 없다."""


class ScheduleTimeExpressionError(ValueError):
    """원문의 시각 표현을 확정된 시각으로 해석할 수 없다."""


@tool
def resolve_schedule_time(expression: str, assume_business_hours: bool = False) -> str:
    """원문의 명확한 시각 표현을 24시간제 HH:MM:SS로 반환한다.

    오전 10시, 오후 3시 30분, 오후 3시 반과 15시 같은 표현을 지원한다.
    H:MM 또는 H:MM:SS는 24시간제로 해석한다.
    시각 뒤의 에, 까지, 부터, 로와 결합 조사를 허용한다. 원문 자체는 변경하지 않는다.
    assume_business_hours가 참이면 오전과 오후가 없는 한글 1~7시는 오후,
    8~11시는 오전, 12시는 정오로 해석한다. 거짓이면 모호한 시각을 거부한다.
    날짜, 확정 여부, 다른 발언을 가리키는 표현은 이 도구가 판단하지 않는다.
    """
    text = expression.strip()
    korean = re.fullmatch(
        r'(?:(?P<period>오전|오후)\s*)?(?P<hour>[0-9]{1,2})\s*시'
        r'(?:\s*(?:(?P<minute>[0-9]{1,2})\s*분|(?P<half>반)))?'
        + r'\s*' + _TEMPORAL_PARTICLE, text,
    )
    clock = re.fullmatch(
        r'(?P<hour>[0-9]{1,2}):(?P<minute>[0-9]{2})(?::(?P<second>[0-9]{2}))?'
        + r'\s*' + _TEMPORAL_PARTICLE, text,
    )
    match = korean or clock
    if match is None:
        raise ScheduleTimeExpressionError('현재 지원하지 않는 시각 표현입니다.')

    hour = int(match['hour'])
    minute = 30 if korean and korean['half'] else int(match['minute'] or 0)
    second = int(clock['second'] or 0) if clock else 0
    period = korean['period'] if korean else None
    valid_hour = 1 <= hour <= 12 if period else 0 <= hour <= 23
    if not valid_hour or not 0 <= minute <= 59 or not 0 <= second <= 59:
        raise ScheduleTimeExpressionError('원문 시각 표현에 유효하지 않은 시각이 포함되어 있습니다.')
    if korean and period is None and 1 <= hour <= 12:
        if not assume_business_hours:
            raise ScheduleTimeExpressionError('오전 또는 오후가 없어 시각을 확정할 수 없습니다.')
        if hour <= 7:
            hour += 12
    if period:
        hour = hour % 12 + (12 if period == '오후' else 0)
    return f'{hour:02d}:{minute:02d}:{second:02d}'


@tool
def resolve_schedule_date(expression: str, meeting_date: str) -> str:
    """원문의 명시적인 날짜 또는 상대 날짜를 YYYY-MM-DD로 반환한다.

    meeting_date는 한국 기준 회의 날짜를 YYYY-MM-DD로 전달한다.
    오늘, 내일, 모레, 다음 주 요일, YYYY년 M월 D일, YYYY-MM-DD를 지원한다.
    오전 10시, 오후 3시 30분 같은 시각이 붙어도 날짜 부분을 계산한다.
    표현 끝의 에, 까지, 부터, 로와 결합 조사를 허용한다. 원문 자체는 변경하지 않는다.
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
        r'(?:(?P<minute>[0-9]{1,2})분|반)?)?' + _TEMPORAL_PARTICLE, normalized,
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
