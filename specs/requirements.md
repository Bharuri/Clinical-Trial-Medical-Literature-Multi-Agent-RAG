Requirements - Clinical Trial & Medical Literature Multi-Agent RAG
1. Purpose
Build a Streamlit application that answers clinical trial and medical literature questions using PDF documents, DuckDB vector search, Sentence Transformers embeddings, and Amazon Bedrock.
2. PDF Ingestion and Chunking
- Read medical literature and clinical trial PDFs.
- Extract narrative text and tables separately.
- Convert each complete table to Markdown and store it as one independent chunk.
- Do not split tables across chunks or mix table rows with narrative text.
- Remove repetitive headers, footers, and page numbers from narrative text.
- Split narrative text recursively into chunks of 500-1000 tokens, with 100-200 tokens of overlap.
- Do not let narrative chunks cross table boundaries.
- Preserve sentence and table meaning.
- Store each chunk as a JSON object with:
  - source_file: source PDF name
  - page_number: page where the content originates
  - chunk_type: text or table
  - section_header: nearest parent heading or section title
  - content: extracted text or Markdown table
3. Embeddings and DuckDB Storage
- Generate embeddings locally using Hugging Face sentence-transformers.
- Use an embedding model that outputs 768-dimensional vectors.
- L2-normalize generated vectors.
- Process text and table chunks in batches.
- Store chunks in a DuckDB table named medical_document_chunks with these columns:
  - id: VARCHAR, unique UUID for each chunk
  - source_file: VARCHAR
  - page_number: INTEGER
  - chunk_type: VARCHAR, either text or table
  - section_header: VARCHAR
  - content: VARCHAR
  - embedding: FLOAT[768]
- Install and load DuckDB's vss extension.
- Validate that each embedding contains exactly 768 dimensions before inserting it.
- Create an HNSW index on the embedding column using cosine distance.
- Use safe parameter binding and batch inserts.
4. Amazon Bedrock Configuration
- Use Amazon Bedrock for the foundation models used by the multi-agent pipeline.
- Accept the following configuration for each user request:
  - aws_access_key_id
  - aws_secret_access_key
  - aws_region_name
  - model_id
- Create the Bedrock runtime client using the credentials and region provided for that request.
5. Query Routing and Retrieval
- Analyze each user question to determine whether it needs narrative context, structured data, or both.
- Generate a 768-dimensional query embedding.
- Search medical_document_chunks in DuckDB for the most relevant chunks.
- Retrieve the top-k relevant chunks, including both narrative text and tables when needed for numerical questions.
- Pass the retrieved content and its source_file, page_number, chunk_type, and section_header metadata to the next agent.
6. Answer Synthesis
- Use the retrieved text and Markdown tables to prepare a clear answer.
- Include an inline citation with the source PDF name and page number for every claim, metric, and data point.
- Use table values accurately without changing their row-to-column alignment.
- Keep answers objective, professional, and scientifically rigorous.
7. Fact-Checking
- Check the draft answer sentence by sentence against the retrieved source chunks.
- Flag claims that are not supported by the retrieved content.
- Verify that citations match the source file and page metadata.
- Check numerical values against the retrieved tables.
- Return is_approved: true when the answer is fully verified.
- Return is_approved: false with a concise critique when corrections are needed, and send the draft back for revision.
8. Streamlit User Interface
- Display the title: 🧬 Clinical Trial & Medical Literature Multi-Agent RAG.
- Display a subtitle describing DuckDB, 768-dimensional Sentence Transformers embeddings, and Amazon Bedrock.
- Add a sidebar section titled 🔐 Secure AWS Configuration with:
  - AWS Access Key ID
  - AWS Secret Access Key
  - AWS Region dropdown
  - Bedrock Model ID, defaulting to Claude 3.5 Haiku
- Mask secret credential input.
- Provide a Clear Credentials / Reset Session button that clears Streamlit session state and reruns the app.
- Provide a text box labeled Enter your clinical trial / medical research question:.
- Provide a Run Multi-Agent Pipeline button.
- Show the pipeline steps in Streamlit containers or expanders:
  1. Router & Retrieval Agent: query vector dimensions, HNSW match count, similarity scores, source file names, page numbers, and content previews.
  2. Synthesis & Drafting Agent: progress indicator and cited Markdown draft.
  3. Critic & Fact-Checking Agent: approval status, verification logs, and checks against source chunks.
  4. Final Verified Output: display the final answer in a separate section.
9. Credential and Session Handling
- Do not write AWS credentials or user queries to local storage, cookies, or cache files.
- Keep credentials and session data in Streamlit's session state.
- Clear credentials and session state when the user uses the reset button.
- The application should use the Streamlit WebSocket session lifecycle for session data.