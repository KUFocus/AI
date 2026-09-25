from summary_schema import ScheduleCandidate


def resolve_schedule_history(decisions: list[dict], input_text: str) -> dict | None:
    """같은 일정으로 묶인 결정들을 원문 순서로 적용해 남아 있는 확정본을 반환한다.

    동일 일정의 묶음과 각 발언의 상태 판정은 호출자가 제공한다.
    반환된 날짜가 회의 기준일과 맞는지는 별도 날짜 검증에서 확인한다.
    """
    ordered = []
    for decision in decisions:
        candidate = ScheduleCandidate.model_validate(
            decision, strict=True, context={'input_text': input_text},
        )
        start = input_text.find(candidate.evidence)
        if input_text.find(candidate.evidence, start + 1) != -1:
            raise ValueError('같은 근거 발언이 원문에 반복되어 결정 순서를 확정할 수 없습니다.')
        ordered.append((start, start + len(candidate.evidence), candidate))

    ordered.sort(key=lambda item: item[0])
    previous_end = 0
    confirmed = None
    for start, end, candidate in ordered:
        if start < previous_end:
            raise ValueError('결정의 근거 구간이 겹쳐 변경 순서를 확정할 수 없습니다.')
        previous_end = end
        if candidate.status == 'confirmed':
            confirmed = candidate
        elif candidate.status == 'cancelled':
            confirmed = None
        # 미확정 제안만으로 기존 확정본을 변경하지 않는다.

    return confirmed.model_dump() if confirmed is not None else None
