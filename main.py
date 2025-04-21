import random
import re
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

def tokenize(file: str) -> list:
    # opening the text file
    with open(f'{file}.txt','r') as file:
        nlpLib = spacy.load("el_core_news_sm")
        tokens = []
        entities = []
        doc = nlpLib(file)
        for token in doc:
            tokens.append((token.text, token.pos))
        for ent in doc.ents:
            entities.apend(ent)

        #print(len(tokens), "Total tokens")
        return (tokens,entities)


tokenize("history.txt")

# # Sample mock function to generate quiz questions from a document
# def generate_questions(document, num_questions=3):
#     sentences = sent_tokenize(document)
#     random.shuffle(sentences)

#     questions = []
#     answers = []

#     for i in range(min(num_questions, len(sentences))):
#         sent = sentences[i]
#         words = sent.split()
#         if len(words) > 5:
#             # Create a basic fill-in-the-blank question
#             blank_word = random.choice(words[1:-1])
#             question = sent.replace(blank_word, "_____")
#             questions.append(question)
#             answers.append(blank_word)
    
#     return questions, answers

# # Function to process and grade user answers
# def grade_quiz(user_answers, correct_answers):
#     results = []
#     for i, (user_ans, correct_ans) in enumerate(zip(user_answers, correct_answers)):
#         is_correct = user_ans.strip().lower() == correct_ans.strip().lower()
#         results.append((i + 1, is_correct, user_ans, correct_ans))
#     return results

# # Main CLI function
# def run_quiz():
#     print("Welcome to the Quiz Generator!")
#     document = input("\nPaste your study text here:\n")

#     num_questions = int(input("\nHow many quiz questions would you like? "))

#     print("\nGenerating quiz...\n")
#     questions, answers = generate_questions(document, num_questions)

#     user_answers = []
#     for i, question in enumerate(questions):
#         print(f"Q{i + 1}: {question}")
#         answer = input("Your answer: ")
#         user_answers.append(answer)

#     print("\nGrading your quiz...\n")
#     results = grade_quiz(user_answers, answers)

#     for res in results:
#         q_num, is_correct, user_ans, correct_ans = res
#         if is_correct:
#             print(f"Q{q_num}: ✅ Correct!")
#         else:
#             print(f"Q{q_num}: ❌ Incorrect. Your answer: '{user_ans}'. Correct answer: '{correct_ans}'")

# if __name__ == "__main__":
#     run_quiz()