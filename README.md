# Financial Chatbot


This project implements a document retrieval and question answering system using FAISS for efficient similarity search and GPT-4 for generating answers based on retrieved documents. It utilizes Sentence Transformers for generating document embeddings and integrates with OpenAI's GPT-4 API for natural language processing tasks.

Document Embeddings: Uses Sentence Transformers to convert text documents into dense vector representations.
FAISS Indexing: Efficiently indexes document embeddings using FAISS for fast similarity search.
GPT-4 Integration: Utilizes OpenAI's GPT-4 API to generate answers based on retrieved documents.
Modular Design: Organized into classes for easy extension and maintenance.

Installation
Prerequisites

    Python 3.11
    pip package manager

## Installation Steps
    pip install -r requirements.txt


## Usage
Ensure all dependencies are installed

Ensure you have OpenAI API_KEY

Running the System

Generate Embeddings (Optional): If not using precomputed embeddings, 
run vector_db.py to generate embeddings for the given datasets.

Put your OpenAI API KEY in the **API_KEY** environmental variable.

Execute the main script **python main.py**
    

## Access the Application

Open your web browser and go to http://localhost:5000

## Using Precomputed Embeddings

If you want to skip the step of generating embeddings, you can use precomputed embeddings. 
Make sure the precomputed embeddings are placed in the correct directory as expected by the application.

## Notes

Ensure that the vector storage script vector_db.py is run separately before running the main script, or use the precomputed embeddings if you want to skip that step.
Adjust paths and environment variables as necessary for your setup.