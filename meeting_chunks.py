from dataclasses import dataclass
from typing import Callable

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class MeetingChunk:
    """문자 위치는 0부터 시작하며 end 위치는 포함하지 않는다."""

    index: int
    start: int
    end: int
    text: str
    token_count: int


class MeetingChunker:
    """LangChain으로 분할하고 임베딩 한도와 원문 위치를 확인한다."""

    def __init__(self, count_tokens: Callable[[str], int], *, max_tokens: int = 512,
                 overlap_tokens: int = 64):
        if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
            raise ValueError('청크의 토큰 한도는 1 이상 512 이하의 정수여야 합니다.')
        if type(overlap_tokens) is not int or not 0 <= overlap_tokens < max_tokens:
            raise ValueError('청크 중복 길이는 0 이상 토큰 한도 미만의 정수여야 합니다.')
        overhead = count_tokens('')
        budget = max_tokens - overhead
        if budget < 1 or overlap_tokens >= budget:
            raise ValueError('접두사와 특수 토큰을 제외한 한도는 중복 길이보다 커야 합니다.')
        self.count_tokens = count_tokens
        self.max_tokens = max_tokens
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=budget, chunk_overlap=overlap_tokens,
            length_function=lambda value: max(0, count_tokens(value) - overhead),
            separators=['\r\n\r\n', '\n\n', '\r\n', '\n', '. ', '? ', '! ', ' ', ''],
            keep_separator='end', strip_whitespace=False,
        )

    def split(self, text: str) -> list[MeetingChunk]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('분할할 회의 원문은 비어 있지 않은 문자열이어야 합니다.')
        parts = self.splitter.split_text(text)
        counts = [self.count_tokens(part) for part in parts]
        if any(count > self.max_tokens for count in counts):
            raise ValueError('분할 결과가 임베딩 토큰 한도를 초과했습니다. 청크 한도를 낮춰 다시 분할해 주세요.')
        spans = self._source_spans(text, parts)
        return [MeetingChunk(index, start, end, part, count)
                for index, (part, count, (start, end)) in enumerate(zip(parts, counts, spans))
                if part.strip()]

    @staticmethod
    def _source_spans(text: str, parts: list[str]) -> list[tuple[int, int]]:
        # 토큰 단위 overlap을 문자 수로 빼지 않는다. 반복 문구도 전체 순서와 범위를 대조한다.
        stack = [(0, -1, 0, [])]
        visited = set()
        while stack:
            index, previous_start, covered_end, spans = stack.pop()
            state = (index, previous_start, covered_end)
            if state in visited:
                continue
            visited.add(state)
            if index == len(parts):
                if covered_end == len(text):
                    return spans
                continue
            part = parts[index]
            lower = max(previous_start + 1, covered_end - len(part) + 1, 0)
            upper = covered_end
            if index == 0:
                lower = upper = 0
            candidates = []
            position = text.find(part, lower)
            while 0 <= position <= upper:
                end = position + len(part)
                candidates.append((index + 1, position, end, spans + [(position, end)]))
                position = text.find(part, position + 1)
            stack.extend(reversed(candidates))
        raise ValueError('청크와 원문의 위치를 일관되게 연결하지 못했습니다.')
