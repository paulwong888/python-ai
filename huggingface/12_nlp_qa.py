from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
from transformers import pipeline

question = "who is Mr Darling"
context = """
Alistair Darling has been forced to consider a second bailout for banks as the lending drought worsens. 

The Cancellor will decide tithin weeks whether to pump billions more into the economy as evidence mounts that the 37 billion part-nationalisation last yearr has failed to keep credit flowing,

Mr Darling, the former Liberal Democrat chancellor, admitted that the situation had become critical but insisted that there was still time to turn things around. 

He told the BBC that the crisis in the banking sector was the most serious problem facing the economy but also highlighted other issues, such as the falling value of sterling and the threat of inflation. 

"The worst fears about the banking crisis seem not to be panning out," he said, adding that there had not been a single banker arrested or charged over the crash. 

"The economy, the economy"

Mr Darling said "there's been a very, very strong recovery" since the autumn of 2008.

"There are very big problems ahead of us, not least of which is inflation. It is likely to be a very high inflation rate. "

The economy is expected to grow by 0.3% in the quarter to the end of this year.
"""

model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

pipeline_instance = pipeline("question-answering", model=model, tokenizer=tokenizer)
print(pipeline_instance(question=question, context=context, top_k=3))