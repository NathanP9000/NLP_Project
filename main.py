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

# region Clustering
################
# Chunk Creation
################
def semantic_chunking_avg(text, similarity_threshold=0.7):
    """
    Goes through text and groups together similar sentences if they are similar. 
    Does not group similar sentences if they have unsimilar sentences in between.
    Start with chunk that has first sentence.
    Repeat following steps
    Compute average embedding of chunk
    Cosine similarity of averageEmbedding to current sentence 
    If greater than 0.7 add current sentence to chunk
    else add chunk to chunk list and start a new chunk with the current sentence

    
    :param text: All sentences in the text.
    :param similarity_threshold: Cosine similarity needs to return 0.7 in order to add a sentence to a chunk.
    :return chunks: 
    """ 
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = sent_tokenize(text)
    sentence_embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]
    current_embeddings = [sentence_embeddings[0]]

    for i in range(1, len(sentences)):
        current_sentence = sentences[i]
        current_embedding = sentence_embeddings[i]

        # Compute average embedding of the current chunk
        avg_embedding = np.mean(current_embeddings, axis=0)

        # Compute similarity between current sentence and average of current chunk
        sim = cosine_similarity([avg_embedding], [current_embedding])[0][0]

        if sim >= similarity_threshold:
            current_chunk.append(current_sentence)
            current_embeddings.append(current_embedding)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [current_sentence]
            current_embeddings = [current_embedding]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


############################
# Get keywords for a cluster
############################
def get_cluster_keywords(texts, labels, num_clusters, topicWords=3):
    cluster_keywords = {}

    # Goes through each cluster and finds each chunk that belongs to it then uses tfid to get {topicWords} words that describe the cluster 
    for cluster_id in range(num_clusters): 
        # Get the sentences that belong to the current cluster
        cluster_texts = [texts[i] for i in range(len(labels)) if labels[i] == cluster_id]

        # Create a TF-IDF vectorizer and fit it to the cluster's texts
        vectorizer = TfidfVectorizer(stop_words="english", max_features=topicWords)
        tfidf_matrix = vectorizer.fit_transform(cluster_texts)

        # Get the top n keywords based on TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        cluster_keywords[cluster_id] = feature_names

    return cluster_keywords

####################
# Clustering chunks
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
    # Retrieve chunks from text
    chunks = semantic_chunking_avg(text, similarity_threshold=0.7)
    
    # Turn chunks into embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)

    # Use KMeans to cluster each chunk
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings) # 0 -> numClusters and there are len(chunks) of them (tells us which cluster each chunk belongs to)
    
    # Put all chunks belonging to a cluster together in chronological order
    topics = defaultdict(list)
    for label, chunk in zip(labels, chunks):
        topics[label].append(chunk)
    
    # Get keywords for a chunk (describes each chunk) - might need to do something more descriptive than this
    cluster_keywords = get_cluster_keywords(chunks, labels, num_clusters)
    print("Cluster Keywords:", cluster_keywords)

    # Write down for each cluster keywords and chunks
    with open("topics_and_keywords.txt", "w") as text_file:
        for key in topics.keys():
            text_file.write(str(cluster_keywords[key]))
            text_file.write(str(topics[key]))
            text_file.write('\n')

    return topics, cluster_keywords  # Return both topics and keywords
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
text_file = open("Output.txt", "w")
# Open the file in read mode ("r")
with open("history.txt", "r", encoding="utf-8") as file:
    # Read the entire content of the file
    text_content = file.read()
    topics, clusterkeys = cluster_topics(text_content)
    #mcqs = generate_all_mcqs(topics, n_questions=5)
    #save_mcqs_to_file(mcqs)
text_file.close()
#endregion