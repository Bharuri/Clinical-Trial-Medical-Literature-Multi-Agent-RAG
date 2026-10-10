Tasks — Clinical Trial & Medical Literature Multi-Agent RAG
Complete the tasks in order. Each task must be independently runnable or testable. Keep all code under src/. The first three implementation steps (PDF ingestion, embeddings, and DuckDB storage) belong to one independently runnable ingestion module. Retrieval and agents belong to a separate retrieval_agent module. The Streamlit application belongs to its own app module. Reusable code, including connections, belongs in src/utils/. Configuration stays at the project root.
1. Project Structure
Create this structure first:
project-root/
├── config.yaml
├── requirements.txt
├── requirements.md
├── design.md
├── task.md
├── data/
│   └── pdfs/                     # Input medical literature PDFs
├── database/                     # Local DuckDB database
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_extraction.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── storage.py
│   │   └── run.py                # Run ingestion independently
│   ├── retrieval_agent/
│   │   ├── __init__.py
│   │   ├── retrieval.py
│   │   ├── bedrock_client.py
│   │   ├── synthesis.py
│   │   ├── fact_check.py
│   │   └── run.py                # Run retrieval/agent flow independently
│   ├── app/
│   │   ├── __init__.py
│   │   └── streamlit_app.py      # Streamlit entry point
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       └── connections.py        # Reusable connection setup
└── tests/
2. Tasks
Task 1 — Create the project structure and configuration
- [ ] Create the folder and file structure above.
- [ ] Add root-level config.yaml with the required settings, including PDF input directory, DuckDB path, embedding model, chunk size, overlap, and default Bedrock model ID.
- [ ] Add requirements.txt for the required PDF processing, Sentence Transformers, DuckDB, AWS Bedrock, Streamlit, and testing packages.
- [ ] Implement src/utils/config.py to load root configuration.
- [ ] Implement src/utils/connections.py for reusable connection setup, including DuckDB and Bedrock client creation where appropriate.
- [ ] Do not store AWS secret keys in config.yaml.
Check: Configuration loads and the project imports successfully. The Streamlit entry point can start.
Task 2 — Implement the independently runnable ingestion module
Folder: src/ingestion/
- [x] Extract PDF narrative text and tables separately.
- [x] Convert each complete table to Markdown and keep it as one chunk.
- [x] Remove repetitive headers, footers, and page numbers from narrative text.
- [ ] Recursively split narrative text into 500–1,000-token chunks with 100–200-token overlap. Do not cross table boundaries. (Not verified: size is measured in whitespace words, not model tokens.)
- [x] Attach source_file, page_number, chunk_type, and section_header metadata to each chunk.
- [x] Generate local Sentence Transformers embeddings for text and table chunks in batches.
- [x] L2-normalize embeddings and validate that each vector has exactly 1,024 dimensions.
- [x] Create the medical_document_chunks DuckDB table with the required columns and FLOAT[1024] embedding.
- [x] Load DuckDB's vss extension, insert chunks using parameter binding and batch inserts, and create the HNSW cosine index.
- [x] Implement src/ingestion/run.py so ingestion can be run without launching Streamlit or the agent pipeline.
Check: Run the ingestion module against a sample PDF and a local DuckDB database. Confirm that text and table chunks, metadata, 1,024-dimensional embeddings, and the index are stored correctly. ✅ VERIFIED (DB query: 231 chunks, 155 text / 76 table, 5 sources, all 1024-dim, no null metadata, HNSW index present; full suite 145 passed)
Task 3 — Implement the independently runnable retrieval and agent module
Folder: src/retrieval_agent/
- [x] Implement retrieval using the same embedding model and the medical_document_chunks DuckDB table.
- [x] Determine whether a query needs narrative text, tables, or both; retrieve the top-k relevant chunks with similarity scores and source metadata. (Router is keyword-based, not LLM-based.)
- [x] Use src/utils/connections.py to create a Bedrock runtime client from request-provided AWS access key, secret key, region, and model ID. (Mock-verified; no live AWS call.)
- [x] Implement synthesis that drafts an objective answer with source PDF and page citations for every claim, metric, and data point. (Scripted-LLM verified; real model compliance not tested.)
- [x] Implement fact-checking for unsupported claims, citations, and numerical values.
- [x] Return is_approved: true for verified drafts; otherwise return is_approved: false with a concise critique and route the draft for revision.
- [x] Implement src/retrieval_agent/run.py so retrieval and agents can be run independently of the Streamlit UI. (Full mode needs AWS credentials, not run.)
Check: Run a sample query against the populated DuckDB database. Test retrieval and use mocked Bedrock responses to verify synthesis, citations, fact-checking, and revision flow. ✅ VERIFIED (54 focused tests passed; full suite 145 passed; `--retrieve-only` run returned 5 matches with 1024-dim query; live Bedrock not run)
Task 4 — Implement the Streamlit application
Folder: src/app/
- [x] Display the application title and subtitle. (AppTest verified)
- [x] Add the AWS configuration sidebar with access key, masked secret key, region dropdown, and Bedrock model ID. (AppTest verified; secret field is PASSWORD type; regions come from config.yaml)
- [x] Add the Clear Credentials / Reset Session button. (AppTest verified: clears inputs and results)
- [x] Add the research question input and Run Multi-Agent Pipeline button. (AppTest verified)
- [x] Call the retrieval/agent module without duplicating its logic in the UI. (`src/app/service.py` calls `run_pipeline`; service tests verified)
- [x] Display retrieval vector dimensions, match count, similarity scores, file names, page numbers, and content previews. (AppTest verified metrics and table row)
- [ ] Display synthesis progress and the cited draft. (Cited draft verified. Progress is a spinner plus revision count after the run; live per-step progress needs a callback in `run_pipeline`, not added. Open.)
- [x] Display fact-checking status and verification logs. (AppTest verified)
- [x] Display the final verified answer separately. (AppTest verified: approved shows final; rejected does not)
- [x] Keep credentials and queries out of local storage, cookies, and cache files. (Only `st.cache_resource` on the embedding model; no file, cookie, or storage writes in `src/app`)
Check: Run Streamlit and complete a query through retrieval, synthesis, fact-checking, and final output. ⏳ NOT VERIFIED END-TO-END: server starts and serves (`/_stcore/health` = ok). Full flow mock-verified via AppTest with fake retriever/LLM. Live Bedrock not run (no AWS credentials).
Task 5 — Verify the complete workflow
- [ ] Run ingestion independently and confirm it populates DuckDB.
- [ ] Run retrieval and agents independently against that database.
- [ ] Run the Streamlit app and confirm it calls the same retrieval/agent code.
- [ ] Run tests and fix failures.
Check: The ingestion module, retrieval/agent module, and app can each be started or tested separately, while the app uses the same stored chunks and agent logic.