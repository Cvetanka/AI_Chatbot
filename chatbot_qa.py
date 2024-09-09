import os
import openai
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gc
import torch
torch.set_num_threads(1)
import pickle
import logging
from loguru import logger
from diskcache import Cache
# import re


logger = logging.getLogger(__name__)

class DocumentRetrievalQA:
    """
    Class for document retrieval and question answering using FAISS and GPT-4.
    """

    def __init__(self, model, device, local_folder, cache_folder='/tmp/cache'):
        """
        Initialize the DocumentRetrievalQA class.

        Args:
            model (SentenceTransformer): The SentenceTransformer model.
            device (torch.device): The device to use for computation.
            local_folder (str): Path to the local folder where indices are stored.
            cache_folder (str): Path to the folder where cache is stored.
        """
        self.model = model
        self.device = device
        self.local_folder = local_folder
        self.cache = Cache(cache_folder)

    def set_openai_api_key(self, api_key):
        """
        Set the OpenAI API key.

        Args:
            api_key (str): OpenAI API key.
        """
        openai.api_key = api_key

    def load_faiss_index(self, company_name):
        """
        Load the FAISS index for a given company.

        Args:
            company_name (str): The name of the company.

        Returns:
            index (faiss.Index): The FAISS index.
            index_to_docstore_id (dict): Mapping of index to document store IDs.
            docstore (dict): The document store.
        """
        index = faiss.read_index(os.path.join(self.local_folder, f'{company_name}_index.index'))
        with open(os.path.join(self.local_folder, f'{company_name}_index_to_docstore_id.pkl'), 'rb') as f:
            index_to_docstore_id = pickle.load(f)
        with open(os.path.join(self.local_folder, f'{company_name}_docstore.pkl'), 'rb') as f:
            docstore = pickle.load(f)
        return index, index_to_docstore_id, docstore

    def generate_query_embedding(self, query):
        """
        Generate query embeddings using SentenceTransformer.

        Args:
            query (str): The query string.

        Returns:
            np.ndarray: The query embedding.
        """
        return self.model.encode([query], convert_to_tensor=True, device=self.device).cpu().numpy()[0]

    def query_gpt4_with_prompt(self, prompt):
        """
        Use GPT-4 to answer questions based on the most similar document.

        Args:
            prompt (str): The prompt string.

        Returns:
            str: The response from GPT-4.
        """
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0
        )
        return response.choices[0].message['content'].strip()

    def generate_subqueries(self, query):
        """
        Generate subqueries for general questions.

        Args:
            query (str): The query string.

        Returns:
            list: A list of subqueries.
        """
        prompt = f"Decompose the following query into relevant subqueries: {query}"
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0  # Set temperature to 0 for deterministic output
        )
        subqueries = response.choices[0].message['content'].strip().split('\n')
        return subqueries

    def extract_company_names(self, query):
        """
        Determine the company names from the query.

        Args:
            query (str): The query string.

        Returns:
            list: A list of company names mentioned in the query.
        """
        company_names = ['Sunshine Gloria', 'Deep Mind', 'Tropical Flowers', 'Moonlight Shadow']
        mentioned_companies = [company for company in company_names if company.lower() in query.lower()]
        return mentioned_companies

    def answer_query(self, query):
        """
        Answer the user query.

        Args:
            query (str): The query string.

        Returns:
            list: A list of answers.
        """
        cache_key = f'query:{query}'
        cached_response = self.cache.get(cache_key)

        if cached_response:
            return cached_response

        mentioned_companies = self.extract_company_names(query)

        if len(mentioned_companies) == 0:
            return ["Company name not found in query."]
        elif len(mentioned_companies) > 1:
            return ["Please ask about only one company at a time."]

        company_name = mentioned_companies[0]

        # Determine if the query is general
        if "report" in query.lower() or "overview" in query.lower() or "summary" in query.lower() or "months" in query.lower():
            subqueries = self.generate_subqueries(query)
        else:
            subqueries = [query]

        try:
            index, index_to_docstore_id, docstore = self.load_faiss_index(company_name)
        except Exception as e:
            logger.error(f"Error loading FAISS index for company '{company_name}': {str(e)}")
            return ["Error loading company data. Please try again later."]
        # Collect answers for subqueries
        answers = []
        for subquery in subqueries:
            try:
                # Generate query embedding
                query_embedding = self.generate_query_embedding(subquery)

                # Perform the search
                D, I = index.search(np.array([query_embedding]), k=1)

                # Retrieve the most similar document
                most_similar_doc_id = index_to_docstore_id[I[0][0]]
                most_similar_doc = docstore[most_similar_doc_id].page_content

                # Construct a prompt for GPT-4
                prompt = (
                    f"Here is some information about the company's financial statement:\n\n"
                    f"{most_similar_doc}\n\n"
                    f"Based on the above document, can you answer the following question? {subquery} "
                    f"Please provide specific details mentioned in the document and avoid including unrelated information."
                    f"Do not state phrases like 'The document does not provide' or 'The document provides'. Do not use the word 'document' or 'documents'."
                    f"If there is no available information for the whole year, please provide the information for the quarters(Q) if available."
                    f"If multiple months are mentioned in the {subquery}, please provide answser for all the months you have available information."

                )


                answer = self.query_gpt4_with_prompt(prompt)
                if answer:  # Append only non-empty answers
                    answers.append(answer.strip())

                # Free memory explicitly
                del query_embedding, D, I, most_similar_doc_id, most_similar_doc, prompt, answer
                gc.collect()

            except Exception as e:
                logger.error(f"Error processing subquery '{subquery}': {str(e)}")
                answers.append("An error occurred while processing your request.")

        self.cache.set(cache_key, answers, expire=4*7*24*60*60)  # Cache the response for 4 weeks
        return answers

