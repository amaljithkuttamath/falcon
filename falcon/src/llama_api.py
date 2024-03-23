import requests
import json

import os


def query_llama_api(question, context=None):
    """
    Sends a POST request to the Llama API with the specified organization, question, and context.

    Parameters:
    - question (str): The question to ask about the context.
    - context (str): The context in which the question is asked.

    Returns:
    - dict: The response from the API.
    """

    # API endpoint
    url = "https://6xz1owomvn04h0-80.proxy.runpod.net/query"

    # Headers to include in the request
    headers = {
        "Authorization": os.environ["APIKEY"],
        "Content-Type": "application/json",
    }

    # Data payload to send in the request
    data = {"org": "falcon", "question": question}
    if context:
        data["context"] = context
        print(context)

    # Make the POST request to the API
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check if the request was successful
    if response.status_code == 200:
        # Return the JSON response if successful
        print(response.json())
        return response.json()["generated_text"]
    else:
        # Handle errors or unsuccessful responses
        return {"error": f"Request failed with status code {response.status_code}"}
