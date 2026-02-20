from langchain_text_splitters import RecursiveCharacterTextSplitter

def custom_split(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(docs)

    # add metadata manually
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i
        doc.metadata["source"] = "AI-NOTES-UNIT-1.pdf"

    return chunks