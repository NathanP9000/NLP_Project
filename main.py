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
import random
import nltk
import requests
nltk.download('punkt_tab')
# Load spaCy model
random.seed(42)
np.random.seed(42)
nlp = spacy.load("en_core_web_sm")

#region clustering
def resolve_pronoun_text(pronoun, context_entities):
    pronoun = pronoun.lower()
    if pronoun in {"they", "them", "their"}:
        return context_entities[-1] if context_entities else pronoun
    elif pronoun in {"he", "him", "his", "she", "her"}:
        return context_entities[-1] if context_entities else pronoun
    elif pronoun in {"it", "its"}:
        return context_entities[-1] if context_entities else pronoun
    return pronoun

def extract_entity_evolution(text):
    doc = nlp(str(text))
    entity_data = defaultdict(lambda: {"Properties": set(), "Relations": []})
    context_entities = []

    RELATION_SUBJ_DEPS = {"nsubj", "nsubjpass", "agent", "expl"}
    RELATION_OBJ_DEPS = {"dobj", "attr", "prep", "pobj", "acomp", "iobj", "xcomp", "ccomp", "relcl", "advcl"}
    
    for sent in doc.sents:
        for token in sent:
            if token.pos_ in {"NOUN", "PROPN"}:
                ent = token.lemma_.lower()
                context_entities.append(ent)
                for child in token.children:
                    if child.dep_ in {"amod", "compound", "poss", "nummod"} or child.pos_ == "ADJ":
                        entity_data[ent]['Properties'].add(child.text.lower())

        for token in sent:
            if token.pos_ == "VERB":
                subj_text, obj_text = None, None
                for child in token.children:
                    if child.dep_ in RELATION_SUBJ_DEPS:
                        subj_text = resolve_pronoun_text(child.text, context_entities) if child.pos_ == "PRON" else child.lemma_.lower()
                    elif child.dep_ in RELATION_OBJ_DEPS:
                        obj_text = resolve_pronoun_text(child.text, context_entities) if child.pos_ == "PRON" else child.lemma_.lower()
                if subj_text:
                    entity_data[subj_text]['Relations'].append((token.lemma_, obj_text))
    return entity_data

#####################################
# Create clusters based on similarity
#####################################
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

    # Track each entity's evolution in a chunk
    cluster_entity_evolution = {}
    for i, topic in enumerate(sorted_topics):
        evolution = extract_entity_evolution(topic)
        cluster_entity_evolution[i] = evolution

    # Write entity evolution to file
    with open("entity_evolution.txt", "w", encoding="utf-8") as f:
        for cluster_id, entities in cluster_entity_evolution.items():
            f.write(f"\n=== Cluster {cluster_id} ===\n")
            for ent, info in entities.items():
                f.write(f"Entity: {ent}\n")
                f.write(f"  Properties: {info['Properties']}\n")
                f.write(f"  Relations: {info['Relations']}\n")
    return topics, cluster_entity_evolution  # Return both topics and keywords
#endregion

#region MCQ
###########################
# Generate MCQ from cluster
###########################
def generate_mcqs_from_cluster(cluster_texts, cluster_entities,cluster_name="Topic", n_questions=5):
    """
    Generates multiple choice questions from a cluster of text chunks using Groq API with requests.

    Args:
        cluster_texts (list[str]): List of chunk strings from a single topic cluster.
        cluster_name (str): Optional label for the topic.
        n_questions (int): Number of MCQs to generate.
    Returns:
        str: A formatted string of multiple choice questions.
    """
    combined_text = "\n".join(cluster_texts)
    
    prompt = f"""
        The following content is a breakdown of entities their properties and relationships between entiies. Analyze what's given and generate {n_questions} multiple choice questions based on it.
        I want one correct answer. One answer that is a distractor and is based off of the properties and entities as well. Then I want one answer that is wrong but is still relevant you can generate this. After creation I want each multiple choice question
        to be briefly cited back to the content I gave you.

        Note that in some of the content that is provided the context of the entities is not obvious. Pick what makes sense for the question.

        Topic: {cluster_name}

        Content:
        \"\"\"
        {cluster_entities}
        \"\"\"
        
        Format:
        Q1. [question]
        A. [option 1]
        B. [option 2]
        C. [option 3]
        D. [correct answer]
        Answer: D
        ---
        Explanation:
        Correct Answer: Explain based on the properties and entities from the document
        Distractor: Explain based on the properties and entities from the document
        (Repeat this format for each question.)
    """

    # Groq API endpoint
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    # Prepare the payload
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You generate multiple choice quiz questions from historical material."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,  # Adjust as needed
        "temperature": 0.7   # Adjust for creativity if needed
    }
    
    # Set headers with API key
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    try:
        # Make the POST request
        response = requests.post(url, json=payload, headers=headers)
        
        # Check for successful response
        response.raise_for_status()
        
        # Parse the JSON response
        response_data = response.json()
        
        # Extract the generated content
        generated_content = response_data["choices"][0]["message"]["content"].strip()
        
        return generated_content
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        return None
    except KeyError as key_err:
        print(f"Error parsing response: {key_err}")
        return None


def generate_random_mcqs(topics_dict,cluster_entityEvolution,num_clusters=5, n_questions=3):
    """
    Randomly selects clusters and generates MCQs for each selected cluster using an LLM.
    
    Args:
        topics_dict (dict[int, list[str]]): Dictionary of clustered topic chunks.
        num_clusters (int): Number of clusters to randomly select.
        n_questions (int): Number of questions per cluster.
    
    Returns:
        dict[int, str]: Mapping of cluster_id to generated MCQs.
    """
    # Step 1: Randomly select clusters
    all_cluster_ids = list(topics_dict.keys())
    
    # Make sure we don't try to select more clusters than exist
    num_clusters = min(num_clusters, len(all_cluster_ids))
    
    # Randomly select cluster IDs
    selected_cluster_ids = random.sample(all_cluster_ids, num_clusters)
    
    # Step 2: Generate MCQs for selected clusters
    mcq_results = {}
    for cluster_id in selected_cluster_ids:
        cluster_texts = topics_dict[cluster_id]
        cluster_entities = cluster_entityEvolution[cluster_id]
        print(f"\nGenerating MCQs for Cluster {cluster_id} ({len(cluster_texts)} chunks)...")
        cluster_name = f"Cluster {cluster_id}"
        
        try:
            mcqs = generate_mcqs_from_cluster(cluster_texts,cluster_entities, cluster_name, n_questions)
            mcq_results[cluster_id] = mcqs
        except Exception as e:
            print(f"Failed to generate MCQs for Cluster {cluster_id}: {e}")
            mcq_results[cluster_id] = f"Error generating MCQs: {str(e)}"
    
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
    mcqs = generate_random_mcqs(topics, cluster_entityEvolution)
    save_mcqs_to_file(mcqs)
#endregion