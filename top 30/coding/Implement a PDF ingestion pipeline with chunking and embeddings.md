pip install pypdf langchain-text-splitters openai tiktoken


import asyncio
import os
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI

# Initialize Async OpenAI Client
# Set your OPENAI_API_KEY environment variable
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"))

class PDFIngestionPipeline:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initializes the pipeline with a recursive tokenizer-based splitter.
        """
        # Using RecursiveCharacterTextSplitter to split by paragraphs, sentences, and words
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Synchronously parses the PDF page-by-page to keep track of metadata.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        reader = PdfReader(pdf_path)
        pages_data = []

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages_data.append({
                    "text": text,
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "page": page_num
                    }
                })
        return pages_data

    def chunk_documents(self, pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits the raw page texts into semantically cohesive chunks, keeping track of page origins.
        """
        chunks = []
        for page in pages_data:
            # Split text on a per-page basis to preserve page-level metadata accuracy
            split_texts = self.text_splitter.split_text(page["text"])
            
            for index, chunk_text in enumerate(split_texts):
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **page["metadata"],
                        "chunk_index": index
                    }
                })
        return chunks

    async def generate_embeddings_batch(self, chunks: List[Dict[str, Any]], batch_size: int = 100) -> List[Dict[str, Any]]:
        """
        Asynchronously requests embeddings from OpenAI in optimal batch configurations.
        """
        embedded_chunks = []
        
        # Process chunks in explicit batches to handle token limits and network efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts_to_embed = [chunk["content"] for chunk in batch]

            try:
                # Non-blocking async API call
                response = await openai_client.embeddings.create(
                    input=texts_to_embed,
                    model="text-embedding-3-small" # 1536-dimensional hyper-efficient model
                )
                
                # Zip the resulting vectors back onto their structured chunk objects
                for j, data in enumerate(response.data):
                    chunk_copy = batch[j].copy()
                    chunk_copy["embedding"] = data.embedding
                    embedded_chunks.append(chunk_copy)
                    
            except Exception as e:
                print(f"Error processing embedding batch starting at index {i}: {e}")
                # In production, implement a robust retry or dead-letter queue mechanism here
                raise e

        return embedded_chunks

    async def run_pipeline(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Orchestrates the entire execution flow of the pipeline.
        """
        print(f"--- Starting Ingestion for: {pdf_path} ---")
        
        # Step 1: Text Parsing
        raw_pages = self.extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(raw_pages)} text-containing pages.")

        # Step 2: Semantic Chunking
        processed_chunks = self.chunk_documents(raw_pages)
        print(f"Generated {len(processed_chunks)} token-bounded semantic chunks.")

        # Step 3: Async Batch Embedding Generation
        final_dataset = await self.generate_embeddings_batch(processed_chunks)
        print(f"Successfully generated embeddings for all chunks.")
        
        return final_dataset

# --- Execution Entrypoint ---
async def main():
    pipeline = PDFIngestionPipeline(chunk_size=400, chunk_overlap=40)
    
    # Replace with a path to your target local PDF file
    sample_pdf = "sample_document.pdf" 
    
    # Create a dummy file if testing locally
    if not os.path.exists(sample_pdf):
        with open(sample_pdf, "wb") as f:
            f.write(b"%PDF-1.4 standard dummy structural data")
        print(f"Created a placeholder file at {sample_pdf}. Replace it with a valid PDF.")
        return

    try:
        enriched_chunks = await pipeline.run_pipeline(sample_pdf)
        
        # Inspecting the structural output format of the first vector item
        if enriched_chunks:
            first_item = enriched_chunks[0]
            print("\n[Sample Structured Output Item]")
            print(f"Metadata: {first_item['metadata']}")
            print(f"Content snippet: {first_item['content'][:60]}...")
            print(f"Embedding Vector Dimensions: {len(first_item['embedding'])}")
            
    except Exception as e:
        print(f"Pipeline Execution Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
