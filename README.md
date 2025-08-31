
# RAG with Llama 2 and LangChain

This project demonstrates **RetrievalAugmented Generation (RAG)** using **Llama 2 (13B, GPTQ quantized model)** and **LangChain**.
RAG improves response accuracy by retrieving relevant context from external documents (PDFs, web pages, etc.) before passing it to the LLM, reducing hallucinations.

The workflow has been tested on **Google Colab with a T4 GPU**.



## 📌 Features

* Loads **Llama 2 13B GPTQ model** from HuggingFace with Transformers.
* Defines **LangChain PromptTemplates** with Llama 2 system prompt format.
* Ingests external knowledge sources:

  * **PDF documents** (e.g., Wikipedia export).
  * **Web pages** (via `UnstructuredURLLoader`).
* Splits documents into chunks for embedding.
* Uses **HuggingFace Embeddings** + **FAISS vector database** (Chroma also supported).
* Builds **RetrievalQA chains** in LangChain:

  * Retrieves relevant context.
  * Prevents hallucinations by instructing the model to only use retrieved context.
* Includes **hallucination checks** (queries outside the knowledge base return "I don’t know").



## 🛠 Installation

Install required dependencies:

```bash
pip install transformers==4.37.2 optimum==1.12.0 autogptq
pip install langchain==0.1.9 langchain_community
pip install sentence_transformers==2.4.0
pip install unstructured unstructuredinference
pip install pdfminer.six==20221105 pdf2image pikepdf==8.13.0 pypdf==4.0.2 pillow_heif==0.15.0
pip install faissgpu==1.7.2
```



## 🚀 Usage

### 1. Load Llama 2

The script initializes **Llama 2 GPTQ** with a HuggingFace pipeline:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
```

### 2. Test the LLM

Run a simple test:

```python
text = "Explain artificial intelligence in a few lines"
result = llm.invoke(prompt.format(text=text))
print(result)
```

### 3. RAG from PDF

* Load and preprocess PDF files.
* Chunk text into manageable pieces.
* Build embeddings + FAISS vector store.
* Query with **RetrievalQA**.

### 4. RAG from Web Pages

* Load URLs with `UnstructuredURLLoader`.
* Split and embed content.
* Query with **RetrievalQA** chain.



## 📊 Example Queries

* *"When was the solar system formed?"*
* *"Explain in detail how the solar system was formed."*
* *"What are the planets of the solar system composed of?"*
* **Hallucination check**: *"How does the transformers architecture work?"* → Model should respond with lack of context.



## 🔮 Future Enhancements

* Add support for more vector databases (Weaviate, Pinecone, Milvus).
* Deploy as an API (FastAPI/Flask).
* Integrate streaming responses with LangChain.



