from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0)

def booking_handler(request: str) -> str:
    return f"Booking flight for request: {request}. Simulated booking action."

def info_handler(request: str) -> str:
    return f"Handling information request: {request}. Simulated information retrieval."

def unclear_handler(request: str) -> str:
    return f"Unable to determine intent for request: {request}. Simulated fallback response."


coordinator_router_prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the user's request and determine which
        specialist handler should process it.
        - If the request is related to booking flights or hotels,
        output 'booker'.
        - For all other general information questions, output 'info'.
        - If the request is unclear or doesn't fit either category,
        output 'unclear'.
        ONLY output one word: 'booker', 'info', or 'unclear'."""),
        ("user", "{request}")
])

coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

chain = {'booker': RunnablePassthrough.assign(output = lambda x: booking_handler(x['request']['request'])),
         'info': RunnablePassthrough.assign(output = lambda x: info_handler(x['request']['request'])),
         'unclear': RunnablePassthrough.assign(output = lambda x: unclear_handler(x['request']['request']))}


delegation_branch = RunnableBranch(
    (lambda x: x['decision'] == 'booker', chain['booker']),
    (lambda x: x['decision'] == 'info', chain['info']),
    chain['unclear']
)

coordinator_agent = {'decision': coordinator_router_chain, 'request': RunnablePassthrough()} | delegation_branch | (lambda x: x['output'])


def main():

    request_a = "Book a flight."
    result_a = coordinator_agent.invoke({'request': request_a})
    print(result_a)


if __name__ == "__main__":
    main()