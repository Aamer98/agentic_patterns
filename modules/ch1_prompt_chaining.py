from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv


load_dotenv()


llm = ChatOpenAI(temperature=0)

extract_prompt = ChatPromptTemplate.from_template("Extract the technical specification from the following text: \n\n {text_input}")

transform_prompt = ChatPromptTemplate.from_template("Transform the following specifications into JSON format with 'cpu', 'ram' and 'gpu' as keys: \n\n{specifications}")


specifications = extract_prompt | llm | StrOutputParser()

transform_chain = {'specifications': specifications} | transform_prompt | llm | StrOutputParser()


input_prompt = "This computer has an Intel i7 with an RTX 6070 with 124 GB of RAM."

result = transform_chain.invoke({'text_input': input_prompt})

print(result)