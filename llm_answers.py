import json
from getpass import getpass
from groq import Groq


# Read the questions, skipping blank lines.
questions = []
with open("text.txt", "r", encoding="utf-8") as file:
    for line in file:
        question = line.strip()
        if question:
            questions.append(question)

if not questions:
    raise ValueError("text.txt has no questions.")

api_key = getpass("Enter your Groq API key: ").strip()
client = Groq(api_key=api_key)
answers = []

for question in questions:
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{
            "role": "user",
            "content": question + "\n\nPlease answer in one clear paragraph of 100-150 words. Do not include tables, code or extra sections.",
        }],
        temperature=0.6,
        max_completion_tokens=1536,
        reasoning_effort="low",
        include_reasoning=False,
    )

    if response.choices[0].finish_reason != "stop":
        raise ValueError("An answer was cut short or did not finish normally. Previous answers are saved.")

    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise ValueError("An answer was empty. Previous answers are saved.")

    answers.append({"question": question, "answer": answer})

    # Save progress after each answer.
    with open("responses.json", "w", encoding="utf-8") as file:
        json.dump(answers, file, indent=4, ensure_ascii=False)

    print("Saved", len(answers), "of", len(questions), "answers")

print("Finished. Answers saved in responses.json")
