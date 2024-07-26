from transformers import pipeline
from langchain import PromptTemplate, LLMChain
import torch

generate_text = pipeline(
    model="aisquared/dlite-v1-355m",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    framework="pt"
)

generate_text("In this chapter, we'll discuss first steps with generative AI in Python.")

template = """Question: {question}
Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=generate_text)

question = "What is electroencephalography?"

print(llm_chain.run(question))