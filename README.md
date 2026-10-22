# ML Training

Weekly sessions for learning to work with AI APIs in Python.

## Project structure

```
test-app/
├── src/test_app/
│   ├── __init__.py
│   ├── main.py      # Summarizer using the Gemini API
│   └── local.py     # Summarizer using a local Ollama model
├── .env.example
├── pyproject.toml
└── README.md
```

## Setup

1. Clone the repo and open the folder.
2. Install dependencies:
```bash
   uv sync
```
3. Copy `.env.example` to `.env` and add your Gemini API key:
```
   GEMINI_API_KEY=your_key_here
   GEMINI_MODEL=gemini-3.8-flash
```
4. Run the Gemini summarizer:
```bash
   uv run python src/test_app/main.py
```
5. For the local version, install [Ollama](https://ollama.com/download), pull a model, and run:
```bash
   ollama pull gemma3:1b
   uv run python src/test_app/local.py
```

## Sessions

### Session 1: Gemini API connection and text summarizer
- Connected to the Gemini API using an API key stored in `.env`.
- Learned the model parameters and what each one does:
  - `temperature`: controls randomness (low = consistent, high = creative).
  - `top_p`: picks the next word from the smallest group of words whose probabilities add up to this value.
  - `top_k`: picks the next word from the top K most likely words only.
  - `max_output_tokens`: maximum length of the response.
  - `stop_sequences`: text that makes the model stop immediately when it appears.
  - `seed`: makes responses more repeatable for the same input.
  - `candidate_count`: number of different responses returned for one prompt.
  - `presence_penalty` and `frequency_penalty`: reduce repeated words and encourage new topics.
  - `system_instruction`: fixed instructions that define the model's role and style.
- Built a simple summarizer: paste text, type `END`, get a short summary.
- Added a retry for `503` server-busy errors.

Branch: `session1`

### Session 1 Assignment: Local model with Ollama
- Checked the machine specs (RAM, GPU, free disk) to choose a model that fits.
- Installed Ollama and ran `gemma3:1b` (about 0.8 GB) from the terminal.
- Connected Python to the local model with the `ollama` library (`local.py`).
- Rebuilt the same summarizer: paste text, type `END`, get a summary, no API key or internet needed.
- Compared the two setups:

| | Gemini | Ollama |
|---|---|---|
| Runs on | Google servers | Local machine |
| Needs | API key + internet | Ollama + enough RAM |
| Max length | `max_output_tokens` | `num_predict` |
| Instructions | `system_instruction` | `system` message |
| Parameters | `GenerateContentConfig(...)` | `options={...}` |
| Response | `response.text` | `response["message"]["content"]` |

- Small local models are less accurate than Gemini, especially on long or Arabic text.

Branch: `session_1_assignment`

## Notes

- Model used locally: `gemma3:1b` (about 0.8 GB), chosen to fit a machine with 8 GB RAM.
