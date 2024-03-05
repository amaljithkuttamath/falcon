import ollama

messages = []

code_prompt = """
Date	Ransom Amount (USD)	Incident Type
2023-01-01	46036	Phishing
2023-01-08	34785	Phishing
2023-01-15	10596	Insider Threat
2023-01-22	23932	Ransomware
2023-01-29	17785	DDoS

            Generate the code <code> for plotting the previous data in plotly,
            in the format requested. The solution should be given using plotly
            and only plotly. Do not use matplotlib.
            Return the code <code> in the following
            format ```python <code>```
        """
messages.append({
            "role": "assistant",
            "content": code_prompt
        })

model = "gemma:2b"
response = ollama.chat(
        model=model,
        messages=messages
        # temperature=temperature,
        # max_tokens=max_tokens,
        # top_p=top_p,
        # frequency_penalty=0,
        # presence_penalty=0,
        # stop=None
    )
from pprint import pprint
pprint(response)