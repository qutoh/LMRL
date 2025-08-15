Shamelessly vibe-coded with Gemini. It apologizes for its error.
### 1. Function
An LLM-driven setting-agnostic narrative generation engine with a terminal-based monitor and separate TCOD window. At its core, the application creates dynamic, interactive stories by simulating a cast of AI-controlled characters within a procedurally generated world. A key design philosophy is the use of unreliable narrators with limited scope on the game - the NPCs are only given the shared context of the story and none of the internal world of the other characters. This naturally (hopefully) creates conflict with other characters that may have competing and secret goals and ambitions, though unfortunately it often just spills this out into the shared context like it's a book, and sometimes it's pretty dumb yeah.

The user can take control of a character to participate directly when it comes to their turn.

<img width="1919" height="1038" alt="Image" src="https://github.com/user-attachments/assets/182d741e-67df-48cd-ab20-a9c84ca5d4b8" /><img width="1549" height="958" alt="image" src="https://github.com/user-attachments/assets/43b67266-704f-4eeb-a2e0-42fb803c4e40" /><img width="1713" height="1045" alt="image" src="https://github.com/user-attachments/assets/4dbaa97c-bb4b-4e77-b612-d18aa43b6d48" />

The system is designed to be highly configurable, managing everything from world creation and character personalities to tile information and level features. Whether or not that does anything to the game is a crapshoot, Gemini likes to randomly drop functions and I'm too damn tired and fogged from RA to do a systematic review for every change.

### 2. Key Features

*   **Procedural World & Level Generation:** The system can generate entire worlds from a simple thematic prompt using a hierarchical "Atlas" system (`atlas_manager.py`). It then generates playable 2D maps for specific scenes using a worldgen method that retains their relational information for providing to NPCs (a 'narrative zoom lens'). Characters created during worldgen to fill scenes persist across other generations, creating a persistent world that does not lose work done by the LLMs developing it.
*   **Scenes:** World state stores and persists scene prompts, selecting a location it fits in (though this can be rather loose) for later reuse. Scenes are coupled to levels, so that the lengthy level generation process is not repeated for it, only when another story scene is given. Atlas can choose where a scene will occur, and the NPCs it generates are inserted into any story that occurs there.
*   **Dynamic AI Cast Management:** An AI agent known as "The Director" actively manages the story's cast. It can introduce new characters from a predefined casting list, generate them from scratch based on context, update their instructions to reflect character development, and remove them from the story when they are no longer relevant, saving them for later use in other stories in the same world.
*   **Specialized AI Agents:** The system uses a suite of specialized AI agents for different tasks, defined in `agents.json`. Besides the Director, these include "Prometheus" for a tool-based dispatcher handling incidental events needing LLM calls, "Atlas" for world-building, and smaller utility agents for tasks like timekeeping and summarization.
*   **Long-Term Memory:** A `Memory Manager` uses a vector database (LanceDB) and sentence embeddings to provide characters with relevant long-term memories, allowing them to recall past events beyond the immediate dialogue context. A maximum context length is dynamically derived from the model(s) loaded by tasks/agents that need the full context vs. short, potentially specialized outputs, and the story is automatically summarized if it passes the safe threshold determined before it.
*   **Structured Command Parsing:** To speed up gameplay, the application frequently prompts the LLMs for 'single-shot' JSON-formatted commands rather than relying solely on prose. A fallback agent parses and repairs potentially malformed LLM outputs, with another fallback to defining needed elements conversationally or with regex'ed keywords. Future plans are to monitor success rate per task to short-circuit to fallbacks when JSON outputs frequently fail validation.
*   **Automated Model Calibration:** The system includes a `Calibration` process that can systematically test LLM tasks to find the optimal temperature and sampling parameters, storing the results for future use to improve response quality. This is slow but dramatically improves response capabilities agnostic to the model chosen, saves time fiddling with temperature/top-P values, and is stored in a JSON file that can be added to forever if you give me what comes out the other end (may not actually improve shit but it uses statistics words so it must be good.)
*   **Individually configurable model use:** Tasks, and users of those tasks (agents, characters, dm's, etc.) can be set to use a specific model instead of the default specified in settings.json. If you have a small model that's good at one thing, it can play too!
*   **Positional Gameplay:** Characters are represented as entities on a 2D tile map (`game_state.py`). Based on movement verbs, an LLM call is made to interpret narrative descriptions of movement to pathfind and update character coordinates, allowing for the little guys to move around (this does nothing :3)
*   **Interactive TUI:** The entire experience is managed through a terminal UI built alongside the `tcod` library.
*   **Bugs:** More bugs than you could ever eat! Get your bugs!
*   **Poor design choices:** Gemini is _dumb_! It needs to be stopped!

### 4. Architectural Choices

*   **Manager-Based Architecture:** The codebase is heavily modularized into distinct `Manager` classes (`StoryEngine`, `UIManager`, `DirectorManager`, `TurnManager`, etc.). This enforces a strong separation of concerns, making the system easier to understand, maintain, and extend? The comments Gemini leaves are often pretty useless lol.
*   **Data-Driven Configuration:** Almost all static data—AI personas, LLM task prompts, game settings, character archetypes, map feature definitions—is externalized into JSON files in the `/data` directory. This allows for extensive modification of the application's behavior without altering the core Python logic. The `Config` class (`config_loader.py`) centralizes access to this data.
*   **Localization Prepared, not Ready:** Though development is rapid and localization is perpetually out of date, the game was built with this in mind from the beginning. Prompts and many system follow typical localization behavior.

### 5. Requirements
In theory, the game can run using any LLM that can handle the lengths of the prompts. I am developing on a 2080 Super 8gb with a 2060 12gb card in the second slot, and it does alright - a 12gb llama model with 22b parameters is handling everything in my personal run folder, I've only kept the second smaller model going during development to make sure nothing breaks when Gemini decides to eat shit.

You want an SSD. You just do. It's time. I am not responsible for crashed heads from model loading thrashing your drives.

In practice many are not capable of the json prompts needed and frequently go to fallback methods, even when calibrated. I will soon be working on being able to disable subsystems gracefully to reduce the time required to getting to see the different characters interact, if nothing else but to speed past worldgen when I want to work on this part of the game.

### 6. Installation/Running
* Install LMStudio and install/copy a model over so that it's available for use.
* Change settings.json, agents.json, and the task_parameters files to use it. Don't worry about agents or tasks that don't have a model specified, these can safely be left blank.
* Parameters overridden by the game for the curious are temperature, top_k (set to 0 to disable), system context, and top_p. All other settings are adjustable in lmstudio.
* Make sure Just in Time (JIT) model loading is enabled if you want to run headless (not open LM Studio directly) or need to save VRAM between worldgen and gameplay. Otherwise load the model(s) you expect it to need in the Developer menu on the left (need to switch to 'Power User' or 'Developer' with the tabs on the bottom to see) The JIT setting is located at the top left next to 'Status: Running' in Settings. Keep in mind JIT will take more time to load between gamestates, it's really better to load them up directly unless you need it.
* pip install -r requirements.txt in the root folder with main.py in a command prompt.
* run main.py. A short wait is normal before the main menu appears with the TCOD window. Keep both windows open - this is necessary because Python still lives in the 13th century for threading. Worldgen will not be possible until models say they are loaded in the console window.
* OPTIONAL but not really I haven't touched temperatures since I implemented it and Gemini is really dumb about setting sensible defaults: If you want things to work well, run Calibration from the main menu.
### 7. Porting
It shouldn't be too much work to move it over to your toaster running gentoo or whatever. Really the only thing especially windows-oriented is the notification library.
### 8. Future
The prompts and general token economy could be vastly improved. Gemini is stupidly wordy and adds redundant shit I didn't ask for everywhere. Fuck this shit.
Right now working on an equipment system that hashes outfits to provide real descriptions of characters. They can't see each other so it only really matters to remind characters they have a sword or already have a hat or something. 
