import os
from openai import OpenAI
from pydantic import BaseModel

API_BASE = "https://lonlie.plus7.plus/v1"
API_KEY = "sk-JhC2NWrNAARa9lbPA388E4250f5c4aE19eB590967c22F9B9"

client = OpenAI(base_url=API_BASE, api_key=API_KEY)

PROMPT = """
Please generate a JSON object with the following fields:
- A field "python_expression", whose value is a Python expression (as a string) that generates a string (for example: ' '.join(str(i) for i in range(10))). The expression must be a valid Python expression that can be directly executed by the eval() function to obtain the string.
- A field "escaped_newline", whose value is the string which contains "\\n", i.e., a string containing an escaped newline character.

Strictly output the following JSON structure:
{
    "python_expression": "...",
    "escaped_newline": "..."
}
"""

class PythonExpressionModel(BaseModel):
    python_expression: str
    escaped_newline: str

    def __str__(self):
        return str(self.python_expression) + "; " + str(self.escaped_newline)

def main():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": PROMPT}
    ]
    try:
        # response = client.beta.chat.completions.parse(
        #     model="o4-mini",
        #     messages=messages,
        #     max_tokens=256,
        #     response_format=PythonExpressionModel
        # )
        response = client.beta.chat.completions.parse(
            model="o4-mini",
            messages=messages,
            max_tokens=8192,
            response_format=PythonExpressionModel,
            reasoning_effort="low",
            verbosity="medium"
        )
        print("Raw response:", response)
        print("Parsed:", response.choices[0].message.parsed)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
