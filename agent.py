from langchain import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd

def create_agent(filename: str, api_key: str):
    """
    Create an agent that can access and use a large language model (LLM).

    Args:
        filename: The path to the CSV file that contains the data.
        api_key: The OpenAI API key.

    Returns:
        An agent that can access and use the LLM.
    """

    # Create an OpenAI object.
    llm = OpenAI(openai_api_key=api_key)

    # Read the CSV file into a Pandas DataFrame.
    df = pd.read_csv(filename)

    # Create a Pandas DataFrame agent.
    return create_pandas_dataframe_agent(llm, df, verbose=False)

def query_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.

    Returns:
        The response from the agent as a string.
    """

    prompt = (
        """
            Please return the response in valid JSON format.

            For a table, the response should look like this:
            {
                "table": {
                    "columns": ["column1", "column2", ...],
                    "data": [[value1, value2, ...], [value1, value2, ...], ...]
                }
            }

            For a bar chart, the response should look like this:
            {
                "bar": {
                    "columns": ["A", "B", ...],
                    "data": [{"A": value1, "B": value2, ...}, {...}, ...]
                }
            }

            For a line chart, the response should look like this:
            {
                "line": {
                    "columns": ["A", "B", ...],
                    "data": [{"A": value1, "B": value2, ...}, {...}, ...]
                }
            }

            For a pie chart, the response should look like this:
            {
                "pie": {
                    "labels": ["A", "B", ...],
                    "values": [value1, value2, ...]
                }
            }

            For a scatter plot, the response should look like this:
            {
                "scatter": {
                    "x": ["A", "B", ...],
                    "y": [value1, value2, ...]
                }
            }

            For a histogram, the response should look like this:
            {
                "histogram": {
                    "bins": ["A", "B", ...],
                    "values": [value1, value2, ...]
                }
            }

            If it is just asking a question that requires neither, reply as follows:
            {
                "answer": "answer"
            }
            Example:
            {
                "answer": "The title with the highest rating is 'Gilead'"
            }

            If you do not know the answer, reply as follows:
            {
                "answer": "I do not know."
            }

            Return all output as a valid JSON string with double quotes for all strings.

            Below is the query.
            Query:
            """
        + query
    )

    # Run the prompt through the agent.
    response = agent.run(prompt)

    # Convert the response to a string.
    return response.__str__()
