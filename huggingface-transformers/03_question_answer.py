from transformers import pipeline

qa = pipeline("question-answering")

olympic_wiki_text = """
The 2020 Summer Olympics (Japanese: 2020年夏季オリンピック, –- Omitted for copyright
reasons.–- Olympic medals.[11][12][13]"""

print(qa(question="What caused Tokyo Olympic postponed?", context=olympic_wiki_text))