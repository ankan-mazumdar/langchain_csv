import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
from agent import query_agent, create_agent

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode response: {e}")
        st.write("Raw response was:")
        st.write(response)
        return None

def write_response(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """
    if response_dict is None:
        return

    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        columns = response_dict["bar"]["columns"]
        data = response_dict["bar"]["data"]
        df = pd.DataFrame(data)
        df.set_index(columns[0], inplace=True)
        st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        columns = response_dict["line"]["columns"]
        data = response_dict["line"]["data"]
        df = pd.DataFrame(data, columns=columns)
        df.set_index(columns[0], inplace=True)
        st.line_chart(df)

    # Check if the response is a pie chart.
    if "pie" in response_dict:
        labels = response_dict["pie"]["labels"]
        values = response_dict["pie"]["values"]
        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

    # Check if the response is a scatter plot.
    if "scatter" in response_dict:
        x = response_dict["scatter"]["x"]
        y = response_dict["scatter"]["y"]
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        ax.set_xlabel("Titles")
        ax.set_ylabel("Ratings Count")
        ax.set_title("Scatter Plot of Titles vs Ratings Count")
        plt.xticks(rotation=90)  # Rotate x labels for better readability
        st.pyplot(fig)

    # Check if the response is a histogram.
    if "histogram" in response_dict:
        bins = response_dict["histogram"]["bins"]
        values = response_dict["histogram"]["values"]
        fig, ax = plt.subplots()
        ax.bar(bins, values)
        ax.set_xlabel("Titles")
        ax.set_ylabel("Ratings Count")
        ax.set_title("Histogram of Titles vs Ratings Count")
        plt.xticks(rotation=90)  # Rotate x labels for better readability
        st.pyplot(fig)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

st.title("👨‍💻 Chat with your CSV")

st.write("Please upload your CSV file below and enter your query.")

api_key = st.text_input("Enter your OpenAI API key:", type="password")
data = st.file_uploader("Upload a CSV")
query = st.text_area("Insert your query")

if st.button("Submit Query", type="primary"):
    if not api_key:
        st.error("Please enter your OpenAI API key.")
    elif not data:
        st.error("Please upload a CSV file.")
    elif not query:
        st.error("Please enter your query.")
    else:
        # Save the uploaded CSV file to a temporary location
        with open("uploaded_file.csv", "wb") as f:
            f.write(data.getbuffer())

        # Create an agent from the CSV file.
        agent = create_agent("uploaded_file.csv", api_key)

        # Query the agent.
        response = query_agent(agent=agent, query=query)

        # Display the raw response for debugging
        st.write("Raw response from the agent:")
        st.write(response)

        # Decode and process the response.
        decoded_response = decode_response(response)
        write_response(decoded_response)
