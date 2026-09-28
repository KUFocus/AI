from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from meeting_index import SearchHit


class MeetingAnswerState(TypedDict, total=False):
    project_id: int
    minutes_id: int | None
    question: str
    hits: list[SearchHit]
    sources: list[dict]
    response_content: str
    result: dict
    abstention_reason: str


def build_meeting_answer_workflow(search, build_context, generate_answer, validate_answer, abstain):
    """재검색이나 재생성 없이 근거 부족과 인용 실패를 보류로 연결한다."""

    def retrieve(state: MeetingAnswerState):
        hits = search(state['project_id'], state['question'],
                      minutes_id=state.get('minutes_id'), top_k=3)
        return {'hits': hits, 'abstention_reason': '' if hits else '검색 결과 없음'}

    def prepare_context(state: MeetingAnswerState):
        sources = build_context(state['project_id'], state['hits'])
        return {'sources': sources, 'abstention_reason': '' if sources else '사용 가능한 근거 문맥 없음'}

    def generate(state: MeetingAnswerState):
        return {'response_content': generate_answer(state['question'], state['sources'])}

    def validate_citations(state: MeetingAnswerState):
        result = validate_answer(state['project_id'], state['response_content'], state['sources'])
        return {'result': result, 'abstention_reason': (
            '' if result['status'] == 'answered' else '모델의 근거 부족 판단 또는 인용 검증 실패'
        )}

    def abstain_answer(state: MeetingAnswerState):
        return {'result': abstain()}

    graph = StateGraph(MeetingAnswerState)
    graph.add_node('retrieve', retrieve)
    graph.add_node('prepare_context', prepare_context)
    graph.add_node('generate', generate)
    graph.add_node('validate_citations', validate_citations)
    graph.add_node('abstain', abstain_answer)
    graph.add_edge(START, 'retrieve')
    graph.add_conditional_edges('retrieve', lambda state: bool(state['hits']),
                                {True: 'prepare_context', False: 'abstain'})
    graph.add_conditional_edges('prepare_context', lambda state: bool(state['sources']),
                                {True: 'generate', False: 'abstain'})
    graph.add_edge('generate', 'validate_citations')
    graph.add_conditional_edges('validate_citations', lambda state: state['result']['status'],
                                {'answered': END, 'insufficient_evidence': 'abstain'})
    graph.add_edge('abstain', END)
    return graph.compile()
