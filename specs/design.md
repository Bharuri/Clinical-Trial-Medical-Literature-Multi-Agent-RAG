Design — Clinical Trial & Medical Literature Multi-Agent RAG
1. System Overview
The application uses Streamlit as the user interface, a local PDF ingestion and embedding pipeline, DuckDB for chunk and vector storage, and Amazon Bedrock for the multi-agent workflow.
The main flow is:
1. The user configures AWS credentials and Bedrock model settings.
2. PDF documents are extracted into text and table chunks.
3. Chunks are embedded locally and stored in DuckDB.
4. The user submits a medical literature or clinical trial question.
5. The router and retrieval agent searches DuckDB for relevant chunks.
6. The synthesis agent drafts an answer with citations.
7. The fact-checking agent verifies the draft against retrieved chunks.
8. The application displays the verified answer.
2. Main Components
2.1 Streamlit UI
Responsibilities:
- Collect AWS configuration and the user's question.
- Provide a session reset control.
- Display retrieval results, synthesis progress, fact-checking results, and the final answer.
2.2 PDF Ingestion and Chunking
Responsibilities:
- Read PDF pages and extract narrative text and tables.
- Convert tables to Markdown and keep each complete table as one chunk.
- Remove repetitive headers, footers, and page numbers from narrative text.
- Recursively split narrative text into chunks of 500–1,000 tokens with 100–200 tokens of overlap.
- Attach source file, page number, chunk type, and section header metadata to every chunk.
Chunk structure:
- source_file
- page_number
- chunk_type
- section_header
- content
2.3 Embedding Generator
Responsibilities:
- Use a local Sentence Transformers model that returns 768-dimensional vectors.
- L2-normalize vectors.
- Generate embeddings in batches for both text and table chunks.
- Validate vector dimensions before storage.
2.4 DuckDB Vector Store
Use a table named medical_document_chunks containing:
- id
- source_file
- page_number
- chunk_type
- section_header
- content
- embedding
The embedding column uses FLOAT[768]. Load DuckDB's vss extension and create an HNSW index using cosine distance.
2.5 Bedrock Client Configuration
For each user request, initialize the Bedrock runtime client using the supplied:
- AWS access key ID
- AWS secret access key
- AWS region
- Bedrock model ID
2.6 Router and Retrieval Agent
Responsibilities:
- Determine whether a question needs narrative content, structured table data, or both.
- Embed the question using the local embedding model.
- Search the DuckDB vector store.
- Retrieve the top-k relevant chunks.
- Preserve source metadata when passing chunks to the synthesis agent.
2.7 Synthesis Agent
Responsibilities:
- Use retrieved chunks to draft a clear answer.
- Cite the source PDF and page number for each claim, metric, and data point.
- Preserve the meaning and numerical alignment of table data.
2.8 Fact-Checking Agent
Responsibilities:
- Check each draft sentence against the retrieved chunks.
- Verify citations and numerical values.
- Return is_approved: true when the draft is supported.
- Return is_approved: false with a concise critique when issues are found, so the draft can be revised.
3. Data Flow
PDF Processing Flow
1. Read a PDF page.
2. Detect narrative text and tables.
3. Extract narrative text and convert tables to Markdown.
4. Split narrative text using the configured chunk size and overlap.
5. Attach metadata to each chunk.
6. Generate and normalize embeddings in batches.
7. Validate that each vector has 768 dimensions.
8. Store chunk content, metadata, and embedding in DuckDB.
Question Answering Flow
1. Receive the user's question from Streamlit.
2. Initialize the Bedrock client with the request's AWS configuration.
3. Generate a query embedding locally.
4. Determine whether narrative chunks, table chunks, or both are needed.
5. Search DuckDB and retrieve the top-k relevant chunks.
6. Display retrieval metadata and content previews.
7. Send retrieved chunks to the synthesis agent.
8. Draft an answer with inline source and page citations.
9. Send the draft and retrieved chunks to the fact-checking agent.
10. If approved, display the final verified answer.
11. If not approved, show the critique and route the draft for revision.
4. DuckDB Data Model
Table: medical_document_chunks
Column	Type	Purpose
id	VARCHAR	Unique chunk identifier
source_file	VARCHAR	Source PDF name
page_number	INTEGER	Source page number
chunk_type	VARCHAR	text or table
section_header	VARCHAR	Nearest section heading
content	VARCHAR	Text or Markdown table
embedding	FLOAT[768]	Normalized chunk embedding


The id value is a UUID. Each row represents one text chunk or one complete table chunk.
5. Agent Handoffs
- Router & Retrieval → Synthesis: retrieved chunk content and metadata, including source file, page number, chunk type, and section header.
- Synthesis → Fact-Checking: drafted answer with inline citations.
- Fact-Checking → Application: approval decision and, if needed, a concise critique for revision.
The fact-checking agent uses the same retrieved source chunks as the synthesis agent to check claims, citations, and numerical values.
6. Streamlit Screen Layout
Sidebar: Secure AWS Configuration
- AWS Access Key ID input.
- Masked AWS Secret Access Key input.
- AWS Region dropdown.
- Bedrock Model ID input, defaulting to Claude 3.5 Haiku.
- Clear Credentials / Reset Session button.
Main Area
- Application title and subtitle.
- Clinical trial or medical research question input.
- Run Multi-Agent Pipeline button.
- Router and retrieval expander with query vector dimensions, match count, similarity scores, file names, page numbers, and content previews.
- Synthesis expander with progress indicator and cited draft.
- Fact-checking expander with approval status and verification logs.
- Separate final verified output section.
7. Session and Credential Handling
- Keep AWS configuration and session data in Streamlit session state.
- Do not write credentials or user queries to local storage, cookies, or cache files.
- Mask the secret key in the UI.
- The reset button clears session state and reruns the application.
- Session data follows the Streamlit WebSocket session lifecycle.
8. Error and Validation Handling
- Validate that every generated embedding has exactly 768 dimensions before insertion.
- Do not insert a chunk whose embedding has an invalid dimension.
- Use parameter binding for database inserts.
- Batch embedding generation and inserts.
- If fact-checking finds unsupported claims, citation errors, or numerical mismatches, return a critique and route the draft for revision rather than marking it approved.