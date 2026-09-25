from pydantic import BaseModel, Field


class ExtractedSchedule(BaseModel):
    extractedScheduleDate: str
    extractedScheduleContent: str


class SummaryResponse(BaseModel):
    summarizedText: str = ''
    schedules: list[ExtractedSchedule] = Field(default_factory=list)
