# inDox : Advance Search and Retrieval Augmentation Generative  

## Overview 

This project combines advanced clustering techniques provided by Raptor with the efficient retrieval capabilities of pgvector. It allows users to interact with and visualize their data in a PostgreSQL database. The solution involves segmenting text data into manageable chunks, enhancing retrieval through a custom model, and providing an interface for querying and retrieving relevant information.

## Prerequisites

Before you can run this project, you need the following installed:
- Python 3.8+
- PostgreSQL
- OpenAI API Key (if using OpenAI embedding model)

Ensure your system also meets the following requirements:
- Access to environmental variables for sensitive information (API keys).
- Suitable hardware to support intensive computational tasks.


## Installation

Clone the repository and navigate to the directory:

```bash
git clone https://github.com/osllmai/inDox.git
cd inDox
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables
Set your `OPENAI_API_KEY` in your environment variables for secure access.

### Database Setup
Ensure your PostgreSQL database is up and running, and accessible from your application.

## Usage

### Preparing Your Data

1. **Define the File Path**: Specify the path to your text or PDF file.
2. **Load Embedding Models**: Initialize your embedding model from OpenAI's selection of pre-trained models.

## Clustering and Retrieval

### Initialize the Retrieval System

```python
from Indox import IndoxRetrievalAugmentation
from langchain_openai import OpenAIEmbeddings
IRA = IndoxRetrievalAugmentation(docs='./sample.txt', collection_name='sample_c',embeddings=OpenAIEmbeddings(), max_tokens=100)
```

### Initialize from other config than the default one

```python
config = {"clustering": {"dim": 10, "threshold": 0.1},
"postgres": {"host": "localhost", "name": "vector_db", "password": "xxx", "port": 5432,
"username": "postgres"}, "qa_model": {"temperature": 0}, "summary_model": {"max_tokens": 100,
"min_len": 30, "model_name": "gpt-3.5-turbo-0125"}, "vector_store": "pgvector"}
IRA = IndoxRetrievalAugmentation.from_config(config=config, docs='./sample.txt', collection_name='sample_c',embeddings=OpenAIEmbeddings(), max_tokens=100)
```
**Note**: You need to change postgres config to your postgres credentials if you set vector_store to pgvector

### Generate Chunks

```python
all_chunks = IRA.get_all_chunks()
print("Chunks:", all_chunks)
```


### PostgreSQL Setup with pgvector

1. **Install pgvector**: To install `pgvector` on your PostgreSQL server, follow the detailed installation instructions available on the official pgvector GitHub repository:

[pgvector Installation Instructions](https://github.com/pgvector/pgvector)

2. **Add Vector Extension**:
   Connect to your PostgreSQL database and run the following SQL command to create the `pgvector` extension:

```sql
-- Connect to your database
psql -U username -d database_name

-- Run inside your psql terminal
CREATE EXTENSION vector;
# Replace the placeholders with your actual PostgreSQL credentials and details
```

### Store in PostgreSQL

```python
# you need to set your database credentials in th config.yaml file
IRA.store_in_vectorstore(all_chunks=all_chunks)
```


### Querying

```python
response, scores, context = IRA.answer_question(query="How did Cinderella reach her happy ending?", top_k=5)
print("Responses:", response)
print("Retrieve chunks:", context)
print("Scores:", scores)
```
### Roadmap

- [ ] vector stores
   - [x] pgvector
   - [x] chromadb  
   - [x] faiss

- [x] summary models
   - [x] openai chatgpt
   - [x] huggingface models

- [ ] embedding models
   - [x] openai embeddings
   - [x] sentence transformer embeddings

- [ ] chunking strategies
   - [x] semantic chunking

- [x] add unstructured support

- [x] add simple RAG support
      
- [ ] cleaning pipeline

- [ ] minor features
   - [x] yaml file


