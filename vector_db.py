import os
from langchain.docstore.document import Document
import faiss
import pickle
import numpy as np
import torch
torch.set_num_threads(1)
from sentence_transformers import SentenceTransformer

import pandas as pd


class CompanyInfoLoader:
    """
    A class to load company information from text files into pandas DataFrames.

    Attributes:
    ----------
    file_paths : list of str
        The file paths to the text files containing the company information.
    dataframes : dict
        A dictionary to store the loaded DataFrames with keys as file names.

    Methods:
    -------
    load_files():
        Loads the text files into DataFrames and stores them in the dataframes attribute.
    get_dataframe(file_name):
        Retrieves the DataFrame for a given file name.
    """

    def __init__(self, file_paths):
        """
        Initializes the CompanyInfoLoader with a list of file paths.

        Parameters:
        ----------
        file_paths : list of str
            The file paths to the text files containing the company information.
        """
        self.file_paths = file_paths
        self.dataframes = {}

    def load_files(self):
        """
        Loads the text files into DataFrames and stores them in the dataframes attribute.
        """
        for file_path in self.file_paths:
            df = pd.read_csv(file_path, sep='delimiter', header=None, engine='python').rename(
                columns={0: 'company_info'})
            file_name = file_path.split("/")[-1]
            self.dataframes[file_name] = df

    def get_dataframe(self, file_name):
        """
        Retrieves the DataFrame for a given file name.

        Parameters:
        ----------
        file_name : str
            The name of the file whose DataFrame is to be retrieved.

        Returns:
        -------
        pd.DataFrame
            The DataFrame corresponding to the given file name.
        """
        return self.dataframes.get(file_name)


file_paths = [
    "text_files/example_1.txt",
    "text_files/example_2.txt",
    "text_files/example_3.txt",
    "text_files/example_4.txt"
]

loader = CompanyInfoLoader(file_paths)
loader.load_files()

# Accessing the dataframes:
text_1_df = loader.get_dataframe("example_1.txt")
text_2_df = loader.get_dataframe("example_2.txt")
text_3_df = loader.get_dataframe("example_3.txt")
text_4_df = loader.get_dataframe("example_4.txt")


class FaissIndexManager:
    """
    A class to manage the creation and storage of FAISS indices for text datasets.

    Attributes:
    ----------
    model : SentenceTransformer
        The sentence transformer model used for generating embeddings.
    device : torch.device
        The device (CPU or GPU) on which the model runs.

    Methods:
    -------
    modify_text(text, company_name):
        Modifies text based on the company name.
    generate_embeddings(texts):
        Generates embeddings for the given texts using the sentence transformer model.
    create_faiss_index(dataset, company_name):
        Creates a FAISS index for a given dataset and company name.
    save_index(company_name, index, index_to_docstore_id, docstore, local_folder):
        Saves the FAISS index and related data to disk.
    create_and_save_indices(datasets, local_folder):
        Creates and saves FAISS indices for multiple datasets.
    """

    def __init__(self):
        """Initializes the FaissIndexManager with a sentence transformer model."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer('multi-qa-mpnet-base-cos-v1').to(self.device)

    def modify_text(self, text, company_name):
        """
        Modifies the text based on the company name.

        Parameters:
        ----------
        text : str
            The text to be modified.
        company_name : str
            The company name to replace or append.

        Returns:
        -------
        str
            The modified text.
        """
        if "The company's" in text:
            return text.replace("The company's", company_name)
        else:
            return company_name + ' ' + text

    def generate_embeddings(self, texts):
        """
        Generates embeddings for the given texts using the sentence transformer model.

        Parameters:
        ----------
        texts : list of str
            The texts for which embeddings are to be generated.

        Returns:
        -------
        np.ndarray
            The generated embeddings.
        """
        return self.model.encode(texts, convert_to_tensor=True, device=self.device)

    def create_faiss_index(self, dataset, company_name):
        """
        Creates a FAISS index for a given dataset and company name.

        Parameters:
        ----------
        dataset : pd.DataFrame
            The dataset containing the texts.
        company_name : str
            The company name to modify the texts.

        Returns:
        -------
        tuple
            A tuple containing the FAISS index, index to docstore ID mapping, and docstore.
        """
        dataset['company_info'] = dataset['company_info'].apply(lambda text: self.modify_text(text, company_name))
        texts = dataset['company_info'].to_list()
        embeddings = self.generate_embeddings(texts).cpu().numpy()
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        docs = [Document(page_content=text) for text in texts]
        index_to_docstore_id = {i: str(i) for i in range(len(docs))}
        docstore = {str(i): docs[i] for i in range(len(docs))}

        return index, index_to_docstore_id, docstore

    def save_index(self, company_name, index, index_to_docstore_id, docstore, local_folder):
        """
        Saves the FAISS index and related data to disk.

        Parameters:
        ----------
        company_name : str
            The name of the company.
        index : faiss.Index
            The FAISS index.
        index_to_docstore_id : dict
            Mapping of index to docstore IDs.
        docstore : dict
            The docstore containing document data.
        local_folder : str
            The path to the local folder where the data will be saved.
        """
        os.makedirs(local_folder, exist_ok=True)
        faiss.write_index(index, os.path.join(local_folder, f'{company_name}_index.index'))
        with open(os.path.join(local_folder, f'{company_name}_index_to_docstore_id.pkl'), 'wb') as f:
            pickle.dump(index_to_docstore_id, f)
        with open(os.path.join(local_folder, f'{company_name}_docstore.pkl'), 'wb') as f:
            pickle.dump(docstore, f)

    def create_and_save_indices(self, datasets, local_folder):
        """
        Creates and saves FAISS indices for multiple datasets.

        Parameters:
        ----------
        datasets : list of tuples
            A list of tuples, each containing a dataset and a company name.
        local_folder : str
            The path to the local folder where the data will be saved.
        """
        indices = {}
        for dataset, company_name in datasets:
            index, index_to_docstore_id, docstore = self.create_faiss_index(dataset, company_name)
            self.save_index(company_name, index, index_to_docstore_id, docstore, local_folder)
            indices[company_name] = (index, index_to_docstore_id, docstore)
        print("Indices created and saved successfully.")
        return indices

    def add_documents(self, docstore_ids, embeddings):
        assert embeddings.shape[1] == self.embedding_dim, \
            f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {embeddings.shape[1]}"

        num_docs = embeddings.shape[0]
        start_idx = len(self.docstore_ids)
        self.index.add(embeddings)

        self.doc_vectors = np.vstack([self.doc_vectors, embeddings])
        self.docstore_ids.extend(docstore_ids)

        return start_idx, start_idx + num_docs

    def update_document(self, docstore_ids, new_embedding):
        doc_idx = self.docstore_ids.index(docstore_ids)
        self.doc_vectors[doc_idx] = new_embedding
        self.index.replace_one(doc_idx, new_embedding)

    def remove_document(self, docstore_ids):
        doc_idx = self.docstore_ids.index(docstore_ids)
        self.doc_vectors = np.delete(self.doc_vectors, doc_idx, axis=0)
        del self.docstore_ids[doc_idx]
        self.index.remove_ids([doc_idx])




datasets = [
    (text_1_df, 'Sunshine Gloria'),
    (text_2_df, 'Deep Mind'),
    (text_3_df, 'Tropical Flowers'),
    (text_4_df, 'Moonlight Shadow')
]

local_folder = "vector_store_1"
index_manager = FaissIndexManager()
indices = index_manager.create_and_save_indices(datasets, local_folder)
