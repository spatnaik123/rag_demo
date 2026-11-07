
from pathlib import Path

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_FOLDER = PROJECT_ROOT / "data" / "documents"

def load_and_chunk_documents():

    all_documents = []

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found.")

    # Step 1: Extract content from PDFs
    for pdf_file in pdf_files:

        print(f"Processing: {pdf_file.name}")

        loader = DoclingLoader(
            file_path=str(pdf_file),
            export_type=ExportType.MARKDOWN
        )

        documents = loader.load()

        # Preserve source filename
        for document in documents:
            document.metadata["source"] = pdf_file.name

        all_documents.extend(documents)

    print(f"\nTotal extracted documents: {len(all_documents)}")

    # Step 2: Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(all_documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


if __name__ == "__main__":

    chunks = load_and_chunk_documents()

    # Display first chunk
    if chunks:
        print("\nFirst chunk:")
        print(chunks[0].page_content)

        print("\nFirst chunk metadata:")
        print(chunks[0].metadata)
