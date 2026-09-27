from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from source_evidence import EvidenceRange


class SummaryTask(BaseModel):
    """담당 업무와 원문을 연결한다. 근거의 존재만 검사하며 수행 약속의 의미는 판단하지 않는다."""

    model_config = ConfigDict(extra='forbid')

    assignees: list[str] = Field(
        description='원문 표기 그대로의 담당자 또는 담당 역할 목록. 공동 담당은 각각 기록. 빈 목록은 담당 정보를 추출하지 않았다는 뜻이며 미정 결정을 의미하지 않음',
    )
    task: str = Field(description='담당 업무의 행동, 대상과 중요한 수행 조건')
    evidence: EvidenceRange = Field(description='담당자와 업무를 함께 뒷받침하는 원문 구간')

    @field_validator('assignees')
    @classmethod
    def validate_assignees(cls, values: list[str]) -> list[str]:
        names = [value.strip() for value in values]
        if any(not name for name in names):
            raise ValueError('담당자 이름은 비어 있거나 공백만으로 이루어질 수 없습니다.')
        if len(names) != len(set(names)):
            raise ValueError('같은 업무에 담당자를 중복해서 기록할 수 없습니다.')
        return names

    @field_validator('task')
    @classmethod
    def validate_task(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('담당 업무 내용은 비어 있거나 공백만으로 이루어질 수 없습니다.')
        return value.strip()

    @model_validator(mode='after')
    def validate_source(self, info: ValidationInfo):
        if not isinstance(info.context, dict) or not isinstance(info.context.get('input_text'), str):
            raise ValueError('담당 업무의 근거 검증에 필요한 회의 원문이 없습니다.')
        source = self.evidence.resolve(info.context['input_text'])
        if not source.strip():
            raise ValueError('담당 업무의 근거 발언은 비어 있을 수 없습니다.')
        for name in self.assignees:
            if name not in source:
                raise ValueError('담당자 이름이 선택한 업무 근거 구간에 존재하지 않습니다. 이름이 적힌 앞선 구간부터 업무 발언까지 함께 포함해 주세요.')
        return self
