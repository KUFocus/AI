import re
from datetime import date, timedelta

from langchain_core.tools import tool


@tool
def resolve_relative_date(expression: str, meeting_date: str) -> str:
    """회의 기준일로 오늘, 내일, 모레, 다음 주 요일을 계산해 YYYY-MM-DD로 반환한다.

    meeting_date는 한국 기준 회의 날짜를 YYYY-MM-DD로 전달한다.
    주의 시작은 월요일이다. 시간이나 일정의 확정 여부는 판단하지 않는다.
    지원하지 않는 표현은 추측하지 않고 오류로 반환한다.
    """
    try:
        reference_date = date.fromisoformat(meeting_date)
        if reference_date.isoformat() != meeting_date:
            raise ValueError('회의 기준일 형식이 올바르지 않습니다.')
    except ValueError as error:
        raise ValueError('회의 기준일은 YYYY-MM-DD 형식의 유효한 날짜여야 합니다.') from error

    normalized = ''.join(expression.split())
    offsets = {'오늘': 0, '내일': 1, '모레': 2}
    if normalized in offsets:
        days = offsets[normalized]
    else:
        match = re.fullmatch(r'다음주([월화수목금토일])요일', normalized)
        if not match:
            raise ValueError('현재 지원하지 않는 상대 날짜 표현입니다.')
        weekday = '월화수목금토일'.index(match.group(1))
        days = 7 - reference_date.weekday() + weekday

    try:
        return (reference_date + timedelta(days=days)).isoformat()
    except OverflowError as error:
        raise ValueError('계산한 날짜가 지원 범위를 벗어났습니다.') from error
