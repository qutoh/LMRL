# /core/summary_manager.py

from core.llm.llm_api import execute_task
from core.common.utils import log_message, count_tokens
from core.common.localization import loc
from core.common import utils


class SummaryManager:
    """A stateful manager for handling story context summarization."""
    def __init__(self, engine):
        self.engine = engine

    def check_and_perform_summary(self):
        """
        Checks if the current prompt context exceeds the token threshold and
        triggers the summarization process if it does.
        """
        if not self.engine.characters: return

        builder = utils.PromptBuilder(self.engine, self.engine.characters[0])
        builder.add_summary(self.engine.summaries).add_dialogue_log(self.engine.dialogue_log)
        
        system_prompt, messages = builder.build_for_token_count()
        prompt_token_count = count_tokens(system_prompt + utils.get_text_from_messages(messages))

        if prompt_token_count > self.engine.token_summary_threshold:
            log_message('debug', loc('system_summary_header', token_count=prompt_token_count, threshold=self.engine.token_summary_threshold))
            if new_summary := self._perform_summary():
                self.engine.summaries.append(new_summary) # Add new summary
                self.engine.dialogue_log.clear() # Clear log after successful summary
                log_message('debug', loc('system_rebuilding_prompt'))

    def _perform_summary(self):
        """Handles the core logic of calling the summarizer agent."""
        dialogue_text = "\n".join(f"{entry['speaker']}: {entry['content']}" for entry in self.engine.dialogue_log)
        previous_summary = self.engine.summaries[-1] if self.engine.summaries else ""
        text_to_summarize = f"{previous_summary}\n\n--- NEW DIALOGUE ---\n\n{dialogue_text}"
        
        summarizer_agent = self.engine.config.agents['SUMMARIZER']
        
        task_key_str = 'SUMMARIZE_MAIN_STORY'
        # Check if a meta-summary is needed due to the length of the *previous* summary
        if count_tokens(previous_summary) > self.engine.token_summary_threshold:
            log_message('debug', loc('system_meta_summary'))
            task_key_str = 'SUMMARIZE_META_STORY'
            text_to_summarize = previous_summary 
            
        new_summary = execute_task(
            self.engine,
            summarizer_agent,
            task_key_str,
            [{"role": "user", "content": text_to_summarize}]
        )
        if new_summary:
            log_message('debug', loc('system_summary_created', summary=new_summary))
            return new_summary
        return None