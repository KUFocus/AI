import json
import logging
from datetime import date, datetime

from summary_schema import SummaryResponse
from summary_workflow import build_summary_workflow


logger = logging.getLogger(__name__)


class MeetingSummarizer:
    def __init__(self, complete, now=datetime.now, *, repair_invalid_response=True):
        self.complete = complete
        self.now = now
        self.workflow = build_summary_workflow(
            self.generate_response, self.validate_response,
            repair_invalid_response=repair_invalid_response,
        )

    def summarize(self, input_text, meeting_date: date | None = None):
        reference_date = meeting_date if meeting_date is not None else self.now().date()
        reference_date_text = reference_date.strftime("%Y-%m-%d")
        messages = [
            {
                "role": "system",
                "content": (
                    f"다음 회의록 내용을 바탕으로 JSON 객체를 만들기 위해 두 가지 작업을 수행해줘. "
                    f"첫째, 회의 내용에서 핵심을 요약하여 'summarizedText'로 반환해줘. 요약에 아래 일정 내용에 적을 일정과 관련된 내용은 절대 포함시키지 마. "
                    f"둘째, 일정이 포함되어 있다면, 각 일정을 'schedules' 리스트로 반환해줘. 일정이 여러 번 언급되더라도, 최종적으로 확정된 일정만 하나씩 반환해줘. "
                    f"각 일정은 'extractedScheduleDate' (LocalDateTime 형식, 예: 2024-11-06T12:49:15), "
                    f"'extractedScheduleContent'로 JSON 객체를 만들어 줘. "
                    f"일정 내용은 '제출', '완성'과 같이 일정표에 적는 것처럼 만들어줘 일정 내용에는 날짜 정보를 절대 포함하시키지 마. "
                    f"일정이 '오늘', '내일', '다음 주', '다음주 목요일'과 같은 상대적인 표현일 경우, 오늘의 날짜({reference_date_text})를 기준으로 해당 날짜를 올바른 LocalDateTime 형식으로 환산해줘. "
                    f"만약 일정이 없다면 빈 리스트로 반환해."
                )
            },
            {"role": "user", "content": input_text}
        ]
        state = self.workflow.invoke({'messages': messages})
        return state['result']

    def generate_response(self, messages) -> str:
        response_content = self.complete(messages)
        logger.info("모델 응답: %s", response_content)
        return response_content

    @staticmethod
    def validate_response(response_content: str) -> dict:
        response_content = response_content.strip()
        if response_content.startswith("```json") and response_content.endswith("```"):
            response_content = response_content[7:-3].strip()

        extracted_data = json.loads(response_content)
        return SummaryResponse.model_validate(extracted_data, strict=True).model_dump()
