import os
import re
import random
import numpy as np
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize
import spacy
import subprocess
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
# Load spaCy model
random.seed(42)
np.random.seed(42)
nlp = spacy.load("en_core_web_sm")


# region Clustering
def extract_entity_evolution(cluster_chunks):
    """
    Given a list of text chunks in a topic cluster, extract named entities
    and show how they evolve (frequency, context) over time.
    Returns: A dictionary where keys are entities and values are their mentions per chunk.
    """
    entity_evolution = defaultdict(list)  # {entity_text: [chunk_0_context, chunk_1_context, ...]}

    for idx, sentence in enumerate(cluster_chunks):
        doc = nlp(sentence)
        # Group entities by text
        entities_in_sentence = defaultdict(list)
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "EVENT"}:  # customize this
                context = ent.sent.text.strip()
                entities_in_sentence[ent.text].append(context)

        # Aggregate into evolution tracker
        for entity, contexts in entities_in_sentence.items():
            entity_evolution[entity].append({
                "sentence_index": idx,
                "mentions": len(contexts),
                "examples": contexts[:3]  # limit context examples for readability
            })

    return entity_evolution

####################
# Clustering sentences
####################
def cluster_topics(text,  num_clusters=30):
    """
    Group the text into chunks and then create clusters for each chunk.
    Chunks in a cluster are grouped together in chronological order and considered a topic.
    Each topic also gets some keywords for descriptive purposes.

    
    :param text: All sentences in the text.
    :param num_cluster: The number of clusters we want. Represents number of topics we expect in the text. 
    :return topic, clusterkeywords: topic maps each cluster to its chunks. Clusterkeywords maps each cluster to its keywords.  
    """ 

    # Tokenize sentences
    sentences = sent_tokenize(text)

    # Turn chunks into embeddings
    model = SentenceTransformer('all-mpnet-base-v2')  # Better for longer chunks
    embeddings = model.encode(sentences)

    # Use KMeans to cluster each chunk
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings) # 0 -> numClusters and there are len(chunks) of them (tells us which cluster each chunk belongs to)
    with open("labels.txt","w") as text_file:
        text_file.write(str(labels))  

    # Put all chunks belonging to a cluster together in chronological order
    topics = defaultdict(list)
    for label, chunk in zip(labels, sentences):
        topics[label].append(chunk)
    sorted_topics = sorted(topics.values(), key=lambda x: len(x), reverse=True)
    with open("topics.txt", "w") as text_file:
        for topic in sorted_topics:
            text_file.write(str(topic))
            text_file.write('\n')   

    # Track each entities evolution in a chunk
    cluster_entity_evolution = {}
    for i, topic in enumerate(sorted_topics):
        evolution = extract_entity_evolution(topic)
        cluster_entity_evolution[i] = evolution

    # Write down for each cluster keywords and chunks
    with open("topics_and_keywords.txt", "w") as text_file:
        for key in topics.keys():
            text_file.write(str(cluster_entity_evolution[key]))
            text_file.write(str(topics[key]))
            text_file.write('\n')

    return topics, cluster_entity_evolution  # Return both topics and keywords
#endregion

#region MCQ
###########################
# Generate MCQ from cluster
###########################
def generate_mcqs_from_cluster(cluster_texts, cluster_name="Topic", n_questions=5):
    """
    Generates multiple choice questions from a cluster of text chunks.

    Args:
        cluster_texts (list[str]): List of chunk strings from a single topic cluster.
        cluster_name (str): Optional label for the topic.
        n_questions (int): Number of MCQs to generate.
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
    print(response)
    return response.choices[0].message.content.strip()


##################################################
# Pass in clusters one by one and ask LLM for MCQs
##################################################
def generate_all_mcqs(topics_dict, n_questions=5):
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
        print(f"\nGenerating MCQs for Cluster {cluster_id} ({len(cluster_texts)} chunks)...")
        cluster_name = f"Cluster {cluster_id}"
        try:
            mcqs = generate_mcqs_from_cluster(cluster_texts, cluster_name, n_questions)
            mcq_results[cluster_id] = mcqs
        except Exception as e:
            print(f"Failed to generate MCQs for Cluster {cluster_id}: {e}")
            mcq_results[cluster_id] = "Error generating MCQs."

    return mcq_results

#########################
# Write down MCQs in file
#########################
def save_mcqs_to_file(mcq_dict, filename="generated_mcqs.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        for cluster_id, mcqs in mcq_dict.items():
            f.write(f"\n===== Cluster {cluster_id} =====\n")
            f.write(mcqs + "\n")
#endregion

#region start
##################
# Start of program
##################        
# Open the file in read mode ("r")
with open("history.txt", "r", encoding="utf-8") as file:
    # Read the entire content of the file
    text_content = file.read()
    topics, cluster_entityEvolution = cluster_topics(text_content)
    #mcqs = generate_all_mcqs(topics, n_questions=5)
    #save_mcqs_to_file(mcqs)
#endregion