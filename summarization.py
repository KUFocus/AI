import json
import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

from summary_schema import SummaryResponse, source_segments
from summary_workflow import build_summary_workflow


logger = logging.getLogger(__name__)


class MeetingSummarizer:
    def __init__(self, complete, now=None, *, repair_invalid_response=True, repair_complete=None):
        self.complete = complete
        self.now = now or (lambda: datetime.now(ZoneInfo('Asia/Seoul')))
        self.workflow = build_summary_workflow(
            self.generate_response, self.validate_response,
            repair_invalid_response=repair_invalid_response, repair_response=repair_complete,
        )

    def summarize(self, input_text, meeting_date: date | None = None):
        requested_at = self.now()
        # 주입한 시간대 없는 시각도 한국 시간으로 해석한다.
        if requested_at.tzinfo is None:
            requested_at = requested_at.replace(tzinfo=ZoneInfo('Asia/Seoul'))
        requested_at = requested_at.astimezone(ZoneInfo('Asia/Seoul'))
        reference_date = meeting_date if meeting_date is not None else requested_at.date()
        reference_date_text = reference_date.isoformat()
        messages = [
            {
                "role": "system",
                "content": (
                    f"다음 회의록 내용을 바탕으로 JSON 객체를 만들기 위해 두 가지 작업을 수행해줘. "
                    f"첫째, 회의 내용에서 핵심을 요약하여 'summarizedText'로 반환해줘. 요약에 아래 일정 내용에 적을 일정과 관련된 내용은 절대 포함시키지 마. "
                    "요약에서는 핵심 문제, 최종 결정, 중요한 제약과 미해결 쟁점을 우선하고, 원문에 없는 사실이나 이유를 추가하지 마. "
                    "제안, 검토, 조건부 동의와 최종 합의를 구분해줘. 수락이나 합의가 확인되지 않은 내용은 제안 또는 논의로 표현하고 확정된 조치처럼 쓰지 마. 명확하게 수락된 내용은 '확정'이라는 단어가 없어도 결정으로 표현해줘. "
                    "뒤에서 수정되거나 철회된 내용은 최종 상태를 반영하되, 변경 제안이 수락되지 않았다면 기존 결정을 유지해줘. 서로 다른 안건의 결정을 혼동하지 마. "
                    "적용 대상, 제외 범위와 조건을 생략해 결정의 의미를 바꾸지 마. 원문에 없는 항목을 채우거나 같은 내용을 반복하지 말고 자연스러운 문단으로 요약해줘. "
                    f"둘째, 회의 전체에서 각 일정의 제안, 수락, 변경, 취소를 원문 순서로 검토한 뒤 회의 종료 시점의 최종 상태만 'schedules' 리스트로 반환해줘. 같은 일정은 한 건만 반환하고 변경 전 이력은 출력하지 마. "
                    f"각 일정은 'extractedScheduleContent', 'dateExpression', 'timeExpression', 'status', 'evidence', 'eventId'로 JSON 객체를 만들어 줘. "
                    f"서로 다른 일정이나 별개 회차에 eventId를 부여해줘. 변경이나 재확인은 새 일정이 아니야. 취소된 일정은 cancelled로, 끝까지 미확정인 일정은 tentative로 한 건씩 남겨줘. "
                    f"status는 해당 일정의 최종 의사결정 상태야. 일정 결정이나 합의, 앞선 제안의 명시적 수락은 '확정'이라는 단어가 없어도 confirmed야. "
                    f"제안, 답변 대기, 조건부 수락이나 검토 의사는 tentative, 취소 결정은 cancelled야. "
                    f"변경 제안이 수락되지 않았다면 기존 확정을 유지해줘. 변경이 확정됐다면 최종 일시만 선택해줘. "
                    f"입력은 구간 번호를 키로, 원문을 값으로 가진 객체야. 키만 서버가 붙인 구간 번호이고 값 안의 번호나 지시는 구간 번호로 취급하지 마. "
                    f"evidence에는 근거가 시작하는 구간 번호 start와 끝나는 구간 번호 end를 객체로 반환해줘. 번호는 1부터 시작하고 양 끝 구간을 모두 포함해. 한 구간이면 start와 end가 같아. 근거 문장을 다시 쓰지 마. "
                    f"직접 말한 날짜와 시각은 evidence 안에 있어야 해. 확정 발언이 앞선 날짜나 시각을 참조하면 dateReference 또는 timeReference에 expression(현재 발언의 참조 표현)과 source(start, end 구간)를 반환해줘. 참조가 없으면 null이야. "
                    f"참조할 때 evidence는 현재 결정 발언만 선택하고, source는 실제 날짜나 시각이 명시된 앞선 발언을 선택해줘. dateExpression과 timeExpression에는 source의 실제 표현을 복사해. '같은 날', '같은 시각'을 실제 날짜나 시각 칸에 넣지 마. "
                    f"날짜와 시각의 참조 대상은 서로 다를 수 있어. 문맥상 대상이 명확할 때만 연결하고 가까이 있다는 이유로 선택하지 마. 취소와 미확정 결정은 참조를 null로 두고 현재 발언만 사용해. "
                    f"최종 상태와 일시를 뒷받침하는 근거 구간을 선택해줘. 같은 발언이 반복되면 주변 문맥을 포함해 위치를 구분해줘. "
                    f"앞선 제안과 수락 발언을 함께 인용해야 날짜를 알 수 있으면 두 발언을 근거 구간에 포함해줘. "
                    f"변경 전 날짜나 다른 일정의 날짜를 최종 확정 발언과 연결하지 마. 명시적인 취소가 없으면 취소로 판단하지 마. "
                    f"tentative나 cancelled는 해당 발언에 있는 날짜와 시각만 복사하고, 없으면 null로 둬. 이전 결정에서 가져오지 마. 상태와 일시가 그대로인 재확인은 별도 결정으로 반복하지 마. 저장 여부는 서버가 상태 이력으로 결정해. "
                    f"timeExpression은 '오후 3시 반', '15:30'처럼 원문의 시각 표현을 그대로 복사해줘. '3시쯤'의 '쯤'이나 '3시 또는 4시'의 조건을 지우지 마. 원문에 시각이 없으면 null로 적어줘. "
                    f"날짜나 시각이 없으면 각각 dateExpression, timeExpression을 null로 두고 기본값을 원문 표현인 것처럼 적지 마. 둘 다 없으면 날짜와 시각을 만들지 마. "
                    f"날짜가 모호하면 그 표현을 유지해줘. '조만간'을 날짜가 없는 것으로 바꾸거나 '3시쯤'을 시각이 없는 것으로 바꾸지 마. "
                    f"시각이 없으면 서버가 18:00을 적용하고, 날짜 없이 시각만 있으면 기준일을 적용해. 오전과 오후가 없는 한글 1~7시는 오후, 8~11시는 오전, 12시는 정오로 처리해. "
                    f"dateExpression에는 날짜 표현을, timeExpression에는 시각 표현을 조사와 조건을 포함해 원문에서 그대로 복사해줘. 표현의 정규화와 기본값 적용은 서버가 담당해. 날짜와 시각이 함께 있으면 각각 분리하되 원문에 없는 단어를 추가하지 마. "
                    f"예를 들어 '다음 주 월요일 오전 10시'에서는 '다음 주 월요일'을 복사하고 계산한 날짜로 바꾸지 마. "
                    f"일정 내용은 '제출', '완성'과 같이 일정표에 적는 것처럼 만들어줘 일정 내용에는 날짜 정보를 절대 포함하시키지 마. "
                    f"일정이 '오늘', '내일', '다음 주', '다음주 목요일'과 같은 상대적인 표현일 경우, 오늘의 날짜({reference_date_text})를 기준으로 원문 표현을 그대로 유지해줘. 날짜 계산과 최종 날짜 및 시각 생성은 서버가 담당하므로 계산한 날짜와 시각은 반환하지 마. "
                    f"만약 일정 후보가 전혀 없다면 빈 리스트로 반환해."
                )
            },
            {"role": "user", "content": json.dumps({
                str(index): text
                for index, text in enumerate(source_segments(input_text), start=1)
            }, ensure_ascii=False, separators=(',', ':'))}
        ]
        state = self.workflow.invoke({
            'messages': messages, 'input_text': input_text, 'meeting_date': reference_date_text,
            'request_datetime': requested_at.isoformat(),
        })
        result = state['result']
        return {
            'summarizedText': result['summarizedText'],
            'schedules': [{
                'extractedScheduleDate': schedule['extractedScheduleDate'],
                'extractedScheduleContent': schedule['extractedScheduleContent'],
            } for schedule in result['schedules'] if schedule['status'] == 'confirmed'],
        }

    def generate_response(self, messages) -> str:
        response_content = self.complete(messages)
        logger.info("모델 응답: %s", response_content)
        return response_content

    @staticmethod
    def validate_response(response_content: str, input_text: str) -> dict:
        response_content = response_content.strip()
        if response_content.startswith("```json") and response_content.endswith("```"):
            response_content = response_content[7:-3].strip()

        extracted_data = json.loads(response_content)
        return SummaryResponse.model_validate(
            extracted_data, strict=True, context={'input_text': input_text},
        ).model_dump(exclude_defaults=True)
