import openai

def ask_chatgpt(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages
    )
    return response["choices"][0]["message"]["content"]

prompt_role = "You are an assistant for journalists. \
    Your task is to write articles, based on the FACTS that are given to you. \
    You should respect the instructions: the TONE, the LENGTH, and the STYLE"

from typing import List

def assist_journalist(
    facts : List[str], tone : str, length_words : int, style : str
):
    facts = ", ".join(facts)
    prompt = f"{prompt_role} \
        FACTS: {facts} \
        TONE: {tone} \
        LENGTH: {length_words} \
        STYLE: {style}"
    print(prompt)
    return ask_chatgpt([{"role" : "user", "content" : prompt}])

print(
    assist_journalist(facts=["The sky is blue", "The grass is green"], tone="informal", length_words=100, style="blogpost")
)