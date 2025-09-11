from openai import OpenAI

API_BASE = "https://lonlie.plus7.plus/v1"
API_KEY="sk-JhC2NWrNAARa9lbPA388E4250f5c4aE19eB590967c22F9B9"

client = OpenAI(base_url=API_BASE, api_key=API_KEY)

response_no = client.responses.create(
    model="o3",
    input="Write a one-sentence bedtime story about a unicorn."
)

response = client.responses.create(
    model="o3",
    input="Write a one-sentence bedtime story about a unicorn.",
    reasoning={
        "effort": "low",
        "summary": "auto"
    }
)

print(response.output_text)
