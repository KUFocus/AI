from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class SummaryState(TypedDict, total=False):
    messages: list[dict[str, str]]
    response_content: str
    result: dict


def build_summary_workflow(generate_response, validate_response):
    def generate(state: SummaryState):
        return {'response_content': generate_response(state['messages'])}

    def validate(state: SummaryState):
        return {'result': validate_response(state['response_content'])}

    graph = StateGraph(SummaryState)
    graph.add_node('generate', generate)
    graph.add_node('validate', validate)
    graph.add_edge(START, 'generate')
    graph.add_edge('generate', 'validate')
    graph.add_edge('validate', END)
    return graph.compile()
