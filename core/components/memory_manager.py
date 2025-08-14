# /core/memory_manager.py

import uuid
from core.common.config_loader import config
from core.common.localization import loc
from core.common.utils import log_message
from core.llm.llm_api import execute_task

try:
    import lancedb
    import pyarrow as pa
    # MODIFIED: SentenceTransformer is no longer imported here
except ImportError as e:
    print(f"FATAL ERROR: Missing required packages for memory management. Please run 'pip install -r requirements.txt'. Details: {e}")
    exit()

class MemoryManager:
    """
    Handles the creation, storage, and retrieval of long-term memories
    using a vector database (LanceDB) and a shared sentence embedding model.
    """
    # MODIFIED: __init__ now accepts the pre-initialized embedding model
    def __init__(self, db_path, embedding_model):
        log_message('debug', loc('log_memory_init_start'))
        if embedding_model is None:
            print(loc('error_memory_init', e="Embedding model was not provided."))
            exit()
        
        try:
            self.model = embedding_model
            db = lancedb.connect(db_path)
            
            embedding_size = self.model.get_sentence_embedding_dimension()
            schema = pa.schema([
                pa.field("vector", pa.list_(pa.float32(), list_size=embedding_size)),
                pa.field("text", pa.string()),
                pa.field("speaker", pa.string()),
                pa.field("id", pa.string()),
                pa.field("timestamp", pa.string())
            ])

            self.table = db.create_table(
                "narrative_memories", 
                schema=schema, 
                mode="create", 
                exist_ok=True
            )
            log_message('debug', loc('log_memory_init_success'))
        except Exception as e:
            print(loc('error_memory_init', e=e))
            exit()

    # ... (the rest of the file is unchanged) ...
    def _summarize_turn_for_storage(self, engine, text_to_summarize: str) -> str:
        """
        Takes a single block of text (one character's action/dialogue) and
        condenses it into a short summary for long-term memory storage using the MNEMOSYNE agent.
        """
        if not text_to_summarize:
            return ""

        mnemosyne_agent = config.agents.get('MNEMOSYNE')
        
        if not mnemosyne_agent:
            log_message('debug', "[MEMORY WARNING] MNEMOSYNE agent not found in agents.json. Storing original text.")
            return text_to_summarize

        summarized_content = execute_task(
            engine,
            mnemosyne_agent,
            'SUMMARIZE_TURN_FOR_MEMORY',
            [{"role": "user", "content": text_to_summarize}]
        )

        if summarized_content and summarized_content.strip():
            log_message('debug', f"[MEMORY] Summarized turn for memory: '{summarized_content.strip()}'")
            return summarized_content.strip()
        else:
            log_message('debug', "[MEMORY WARNING] MNEMOSYNE summarization failed. Storing original text.")
            return text_to_summarize

    def save_memory(self, engine, dialogue_entry):
        """
        Saves a single dialogue entry to the vector database.
        The entry's content is summarized, converted to an embedding, and stored.
        """
        try:
            content = dialogue_entry.get("content", "").strip()
            speaker = dialogue_entry.get("speaker", "Unknown")
            timestamp = dialogue_entry.get("timestamp", "")

            if not content or speaker in ["Scene Setter", "System"]:
                return

            memory_text = self._summarize_turn_for_storage(engine, content)
            document_text = f"{speaker}: {memory_text}"
            vector = self.model.encode(document_text)

            self.table.add([{
                "vector": vector,
                "text": document_text,
                "speaker": speaker,
                "id": str(uuid.uuid4()),
                "timestamp": timestamp
            }])
        except Exception as e:
            log_message('debug', loc('log_memory_save_failed', e=e))

    def retrieve_memories(self, dialogue_log: list, summaries: list):
        """
        Retrieves memories relevant to the most recent context.
        Prioritizes the latest summary for querying, then recent dialogue.
        """
        try:
            if (not dialogue_log and not summaries) or len(self.table) == 0:
                return []

            query_context = ""
            if summaries:
                query_context = summaries[-1]
            elif dialogue_log:
                query_context = "\n".join(
                    f"{entry['speaker']}: {entry['content']}" for entry in dialogue_log[-3:]
                )
            
            if not query_context:
                return []

            count = config.settings.get('MEMORY_RETRIEVAL_COUNT', 3)
            
            n_results = min(count, len(self.table))
            if n_results == 0: return []
            
            query_vector = self.model.encode(query_context)

            results = self.table.search(query_vector).limit(n_results).to_list()
            
            retrieved_docs = [{"text": item['text'], "timestamp": item['timestamp']} for item in results]

            if retrieved_docs:
                log_message('debug', loc('log_memory_retrieved', memories=[doc['text'] for doc in retrieved_docs]))
                return retrieved_docs
            
            return []
        except Exception as e:
            log_message('debug', loc('log_memory_retrieval_failed', e=e))
            return []