# AI Training

Weekly sessions for learning to work with AI APIs in Python.

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
4. Run:
```bash
   uv run python src/test_app/main.py
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
