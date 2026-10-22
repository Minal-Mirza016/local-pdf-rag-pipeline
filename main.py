# ---------------- INDEXING PHASE ----------------

#libraries
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


#DATA LOADING 
#pdf path
PDF_PATH = "doc/AI-Agents-Your-New-Smart-Colleagues.pdf"
#create a pdf loader
loader= PyPDFLoader(PDF_PATH)
#extracting the pdf into page level documents
pages = loader.load()
#display some info
print("pdf loaded succcessfully")
print("total pages:",len(pages))
print("\nFirst page text:")
print(pages[0].page_content[:1000])
print("\nFirst page meta data:")
print(pages[0].metadata)


#CHUNKING
#creating a text splitter
text_splitter= RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
#splitting the pages into chunks
chunks=text_splitter.split_documents(pages)
#display some chunk info
print("\nTotal chunks:", len(chunks))
print("\nFirst chunk")
print(chunks[0].page_content)
print("\nFirst chunk metadata")
print(chunks[0].metadata)
#verifyig the chunks
print("\n--------------------")
for i in range(min(5, len(chunks))):
    print(f"\nChunk {i+1}")
    print(f"Page: {chunks[i].metadata['page']+1}")
    print(chunks[i].page_content)
    print("-"*40)


# DATA EMBEDDINGS
#Creating model for embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#DATA STORING
#creating vector database
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)
#verifying the DB
print("\nVector database created successfully!")
print("Stored chunks:", db._collection.count())


# ---------------- RETRIEVAL PHASE ----------------

# 1. Ask a question about the PDF
query=input("enter the question:")

# 2. Search the vector database for the 3 most relevant chunks
results = db.similarity_search_with_score(query, k=3)

# 3. Display the retrieved chunks
print("\nUSER QUESTION:")
print(query)

print("\nRETRIEVED CHUNKS:")

for i, (document, score) in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print("Page:", document.metadata.get("page", 0) + 1)
    print("Distance score:", round(score, 4))
    print("Text:")
    print(document.page_content)
    print("-" * 50)



# ---------------- GENERATION PHASE ----------------c

# 1. Combine the retrieved chunks into one context
context_text = "\n\n".join(
    document.page_content
    for document, score in results
)

# 2. Define the instructions for the LLM
PROMPT_TEMPLATE = """
You are a helpful document assistant.

Answer the question using only the information
provided in the context.

If the answer is not present in the context,
say that you could not find the answer in the document.

Context:
{context}

Question:
{question}

Answer clearly and accurately.
"""

# 3. Create the prompt template
prompt_template = ChatPromptTemplate.from_template(
    PROMPT_TEMPLATE
)

# 4. Insert the retrieved text and question
messages = prompt_template.format_messages(
    context=context_text,
    question=query
)

# 5. Connect to the local LLM
model = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# 6. Generate the answer
response = model.invoke(messages)

print("\nFINAL AI ANSWER:")
print(response.content)
