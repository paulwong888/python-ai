from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier(
    [
        "Only those who will risk going too far can definitely find out how far one can go.",
        "Baby shark, doo doo doo doo doo doo, Baby shark!"
    ]
)
print(result)