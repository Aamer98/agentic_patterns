import asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, Runnable

load_dotenv()

llm = ChatOpenAI(temperature=0)

summary_chain: Runnable = (ChatPromptTemplate.from_messages(
    [("system", "Generate a breif summary of the given topic."),
    ("user", "{topic}")]
) | llm | StrOutputParser())

keywords_chain: Runnable = (ChatPromptTemplate.from_messages(
    [("system", "Generate three keywords related to this topic."),
    ("user", "{topic}")]
) | llm | StrOutputParser())

questions_chain: Runnable = (ChatPromptTemplate.from_messages(
    [("system", "Generate three questions related to this topic."),
    ("user", "{topic}")]
) | llm | StrOutputParser())

parallel_chain = RunnableParallel(
    {"summary": summary_chain, "keywords": keywords_chain, 
     "questions": questions_chain, "topic": RunnablePassthrough()}
)

synthesis_prompt = ChatPromptTemplate.from_messages(
    [("system", """Given the following:
    Summary: {summary}\n
    Questions: {questions}\n
    Keywords: {keywords}\n
    Generate a comprehensive answer."""),
    ("user", "Original topic: {topic}")]
)

full_parallel_chain = parallel_chain | synthesis_prompt | llm | StrOutputParser()


async def run_parallel_example(topic: str):
    result = await full_parallel_chain.ainvoke({"topic": topic})
    print(result)


if __name__ == "__main__":
    topic = "The impact of artificial intelligence on society."
    asyncio.run(run_parallel_example(topic))