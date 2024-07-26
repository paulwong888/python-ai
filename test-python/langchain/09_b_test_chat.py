import openai

def chat_completion(prompt, model="gpt-4", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role" : "user", "content" : prompt}],
        temperature=temperature
    )
    print(response["choices"][0]["message"]["content"])

prompt = """
Role: You are a nutritionist designing healthy diets for high-performance 
athletes. You take into account the nutrition needed for a good recovery.
Context: I do 2 hours of sport a day. I am vegetarian, and I don't like green 
vegetables. I am conscientious about eating healthily.
Task: Based on your expertise defined in your role, give me a suggestion for a 
main course for today's lunch. With this suggestion, I also want a table with
two columns where each row in the table contains an ingredient from the main 
course. The first column in the table is the name of the ingredient.
The second column of the table is the number of grams of that ingredient needed
for one person. Do not give the recipe for preparing the main course.
"""
chat_completion(prompt)