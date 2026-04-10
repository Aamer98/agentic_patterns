from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(temperature=0.1)

def reflection_agent():

    task_prompt = """
    Your task is to create a Python function named
    `calculate_factorial`.
    This function should do the following:
    1. Accept a single integer `n` as input.
    2. Calculate its factorial (n!).
    3. Include a clear docstring explaining what the function does.
    4. Handle edge cases: The factorial of 0 is 1.
    5. Handle invalid input: Raise a ValueError if the input is a negative """


    # The reflection loop
    max_iterations = 3
    current_code = ""
    message_history = [HumanMessage(content=task_prompt)]


    for i in range(max_iterations):
        
        if i == 0:
            response = llm.invoke(message_history)
            current_code = response.content
        else:
            message_history.append(HumanMessage(content="Please refine the code using the critiques provided"))
            response = llm.invoke(message_history)
            current_code = response.content
        
        reflection_prompt = """
                You are a senior software engineer and an expert
                in Python.
                Your role is to perform a meticulous code review.
                Critically evaluate the provided Python code based
                on the original task requirements.
                Look for bugs, style issues, missing edge cases,
                and areas for improvement.
                If the code is perfect and meets all requirements,
                respond with the single phrase 'CODE_IS_PERFECT'.
                Otherwise, provide a bulleted list of your critiques.
                """
        
        reflection = [SystemMessage(content=reflection_prompt), HumanMessage(content=f'Here is the current code:\n{current_code} and the original task was:\n{task_prompt}')]

        response = llm.invoke(reflection)
        critique = response.content

        if 'CODE_IS_PERFECT' in critique:
            print("Final code is perfect!")

            break
        
        message_history.append(critique)

    print('SOLUTION: \n\n')
    print(current_code)

if __name__=="__main__":
    reflection_agent()