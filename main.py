import random
import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize
import subprocess
import os
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from openai import OpenAI
import os
text_file = open("Output.txt", "w")

print(os.getenv("GROQ_API_KEY"))
client = OpenAI(base_url="https://api.groq.com/openai/v1"
                ,api_key=os.getenv("GROQ_API_KEY"))

# response = client.chat.completions.create(
#     model="llama-3.3-70b-versatile",
#     messages=[
#         {"role": "user", "content": "Can you write a history quiz talking about World War 2 . 5 question mc!"}
#     ]
# )
def cluster_topics(text, chunk_size=3, num_clusters=30):
    # Step 1: Break text into sentence chunks
    sentences = sent_tokenize(text)
    chunks = [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
    
    # Step 2: Convert chunks to sentence embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    # Step 3: Cluster embeddings into topic groups
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    # Step 4: Group chunks by topic label
    topics = defaultdict(list)
    for label, chunk in zip(labels, chunks):
        topics[label].append(chunk)
    
    # Call the function to get cluster keywords
    cluster_keywords = get_cluster_keywords(chunks, labels, num_clusters)
    print("Cluster Keywords:", cluster_keywords)

    # Optional: You can save the topics and keywords to a text file
    with open("topics_and_keywords.txt", "w") as text_file:
        for key in topics.keys():
            text_file.write(str(cluster_keywords[key]))
            text_file.write(str(topics[key]))
            text_file.write('\n')

    return topics, cluster_keywords  # Return both topics and keywords


# Function to extract keywords for each cluster
def get_cluster_keywords(texts, labels, num_clusters, n_keywords=3):
    cluster_keywords = {}

    for cluster_id in range(num_clusters):
        # Get the sentences that belong to the current cluster
        cluster_texts = [texts[i] for i in range(len(labels)) if labels[i] == cluster_id]

        # Create a TF-IDF vectorizer and fit it to the cluster's texts
        vectorizer = TfidfVectorizer(stop_words="english", max_features=n_keywords)
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)

        # Get the top n keywords based on TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        cluster_keywords[cluster_id] = feature_names

    return cluster_keywords


def generate_mcqs_from_cluster(cluster_texts, cluster_name="Topic", n_questions=5, model="gpt-4"):
    """
    Generates multiple choice questions from a cluster of text chunks.

    Args:
        cluster_texts (list[str]): List of chunk strings from a single topic cluster.
        cluster_name (str): Optional label for the topic.
        n_questions (int): Number of MCQs to generate.
        model (str): The LLM model to use (e.g., 'gpt-4' or 'gpt-3.5-turbo').

    Returns:
        str: A formatted string of multiple choice questions.
    """
    combined_text = "\n".join(cluster_texts)
    
    prompt = f"""
        You are an expert history teacher. Read the following content and generate {n_questions} multiple choice questions based on it.
        Each question should have one correct answer and three plausible distractors.
        Topic: {cluster_name}

        Content:
        \"\"\"
        {combined_text}
        \"\"\"

        Format:
        Q1. [question]
        A. [option 1]
        B. [option 2]
        C. [option 3]
        D. [correct answer]
        Answer: D
        ---
        (Repeat this format for each question.)
        """

    client = OpenAI(base_url="https://api.groq.com/openai/v1"
                    ,api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You generate multiple choice quiz questions from historical material."},
            {"role": "user", "content": prompt}
        ]
    )

    return response['choices'][0]['message']['content']


def generate_all_mcqs(topics_dict, model="gpt-4", n_questions=5):
    """
    Sorts clusters by size and generates MCQs for each cluster using an LLM.

    Args:
        topics_dict (dict[int, list[str]]): Dictionary of clustered topic chunks.
        model (str): LLM model to use.
        n_questions (int): Number of questions per cluster.

    Returns:
        dict[int, str]: Mapping of cluster_id to generated MCQs.
    """
    # Step 1: Sort clusters by descending size
    sorted_clusters = sorted(topics_dict.items(), key=lambda x: len(x[1]), reverse=True)

    mcq_results = {}
    for cluster_id, cluster_texts in sorted_clusters:
        print(f"\n📚 Generating MCQs for Cluster {cluster_id} ({len(cluster_texts)} chunks)...")
        cluster_name = f"Cluster {cluster_id}"
        try:
            mcqs = generate_mcqs_from_cluster(cluster_texts, cluster_name, n_questions, model)
            mcq_results[cluster_id] = mcqs
        except Exception as e:
            print(f"⚠️ Failed to generate MCQs for Cluster {cluster_id}: {e}")
            mcq_results[cluster_id] = "Error generating MCQs."

    return mcq_results



# Open the file in read mode ("r")
with open("history.txt", "r", encoding="utf-8") as file:
    # Read the entire content of the file
    text_content = file.read()
    topics = cluster_topics(text_content)
    generate_a;;_mcqs
text_file.close()