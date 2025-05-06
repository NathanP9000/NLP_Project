
# NLP Project: History Quiz Generator

Welcome! This repository contains a Natural Language Processing application that processes a text file to cluster sentences into topics, tracks the evolution of entities across clusters, and generates multiple-choice questions based on the clustered content. The project uses advanced NLP techniques, including sentence embeddings, entity recognition, and integration with the Groq API for question generation, to analyze and summarize historical or textual data (e.g., from `history.txt`).

# Report
https://docs.google.com/document/d/1EjMQ2aIJb6qrNYKEzcZV0XeclVjE8ovlYg8W-gP1IIU/edit?usp=sharing

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Topic Clustering**: Groups sentences from a text file into topics using K-Means clustering and sentence embeddings (via `SentenceTransformer`).
- **Entity Evolution Tracking**: Analyzes entities in each cluster, capturing their properties (e.g., adjectives) and relationships (e.g., subject-verb-object triples) using SpaCy.
- **MCQ Generation**: Generates multiple-choice questions for selected clusters using the Groq API, with correct answers, distractors, and explanations based on entity data.
- **Output Files**: Saves clustering results (`labels.txt`, `topics.txt`), entity evolution (`entity_evolution.txt`), and generated MCQs (`generated_mcqs.txt`).
- Modular code structure for easy extension and customization.

## Prerequisites
- Python 3.8 or higher
- Git (to clone the repository)
- A virtual environment tool (e.g., `venv` or `virtualenv`)
- A Groq API key (sign up at [https://groq.com](https://groq.com) and set it as an environment variable)
- Internet access for downloading NLTK data and accessing the Groq API

## Installation
To set up the project, follow these steps. I strongly recommend using a virtual environment to manage dependencies and avoid conflicts with other projects.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NathanP9000/NLP_Project.git
   cd NLP_Project
   ```

2. **Create and activate a virtual environment**:
   - On Windows:
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   The project includes a `requirements.txt` file listing all required packages. Install them using:
   ```bash
   pip install -r requirements.txt
   ```
   This installs key libraries such as `numpy`, `spacy`, `sentence-transformers`, `scikit-learn`, `nltk`, and `requests`.

4. **Set up SpaCy model**:
   Download the SpaCy English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up LLM API key**:
   Obtain a Groq API key from [https://groq.com](https://groq.com) and set it as an environment variable:
   - On Windows:
     ```bash
     set GROQ_API_KEY=your-api-key
     ```
   - On macOS/Linux:
     - **Option 1: Use `export` command** (temporary, lasts for the current terminal session):
       ```bash
       export GROQ_API_KEY=your-api-key
       ```
     - **Option 2: Use an `env.sh` file** (persistent across sessions):
       Create a file named `env.sh` in the project root:
       ```bash
       echo "export GROQ_API_KEY=your-api-key" > env.sh
       ```
       Source the file to apply the environment variable:
       ```bash
       source env.sh
       ```
       To make it persistent, source `env.sh` in your shell configuration file (e.g., `~/.bashrc` or `~/.zshrc`) by adding `source /path/to/NLP_Project/env.sh`.   

## Usage
1. **Prepare input data**:
   - Place your input text file (e.g., historical or narrative text) in the project root as `history.txt`. Ensure it is encoded in UTF-8.
   - The text will be tokenized into sentences and processed for clustering and MCQ generation.

2. **Run the project**:
   - Execute the main script to process `history.txt`, cluster topics, track entity evolution, and generate MCQs:
     ```bash
     python main.py
     ```
   - The script performs the following:
     - Clusters sentences into topics using K-Means (default: 30 clusters).
     - Extracts entity properties and relationships for each cluster.
     - Generates MCQs for a random subset of clusters (default: 5 clusters, 3 questions each) using the Groq API.
     - Saves results to output files.

3. **View results**:
   - **Clustering outputs**:
     - `labels.txt`: Cluster labels for each sentence.
     - `topics.txt`: Sentences grouped by cluster.
   - **Entity evolution**:
     - `entity_evolution.txt`: Properties and relationships of entities in each cluster.
   - **MCQs**:
     - `generated_mcqs.txt`: Generated multiple-choice questions with answers and explanations.
   - All output files are saved in the project root.

## Project Structure
```
NLP_Project/
├── history.txt          # Input text file (e.g., historical or narrative data)
├── labels.txt           # Output: Cluster labels for sentences
├── topics.txt           # Output: Sentences grouped by cluster
├── entity_evolution.txt # Output: Entity properties and relationships per cluster
├── generated_mcqs.txt   # Output: Generated multiple-choice questions
├── main.py              # Main script to run the project
├── requirements.txt     # List of dependencies
└── README.md            # Project documentation
```

