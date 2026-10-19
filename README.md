# Local Cybersecurity LLM

A local cybersecurity assistant using llama.cpp, Phi-4-mini, and
hybrid retrieval over a cybersecurity skills catalog.

The backend selects relevant skill instructions, adds them to the
model prompt, and checks the input-token budget before generation.

## Architecture

The project runs two Docker services:

- **llama:** Serves the language model through llama.cpp.
- **analyst:** Hosts the browser UI, HTTP API, skill router, and
  context-budget enforcement.

Request flow:

1. The browser submits a prompt to the analyst.
2. The router scores skill metadata using BM25 and MiniLM similarity.
3. The backend loads the highest-ranked skills' instructions.
4. llama.cpp counts the complete input tokens.
5. If the input fits, the backend requests a response.
6. If it exceeds the budget, the UI asks which skills to remove.

The router combines lexical and semantic scores across the catalog.
It does not use BM25 as a preliminary semantic-search filter.

The analyst container uses `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`
as its base image, providing Python 3.12 and uv for dependency installation.
It installs CPU-only PyTorch for the MiniLM embedding model.

## Requirements

- Docker Engine with Docker Compose GPU support.
- An NVIDIA GPU and NVIDIA Container Toolkit for the current CUDA configuration.
- Enough RAM, VRAM, and disk space for the selected models and context size.

Resource requirements depend on the model and inference configuration.

## Clone the project

Clone with submodules to include the cybersecurity skills catalog:

```bash
git clone --recurse-submodules <YOUR_REPOSITORY_URL>
cd cyber-llm
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Models

Model files are excluded from Git and must be obtained separately.

The current configuration uses:

- `Phi-4-mini-instruct-Q4_K_M.gguf` for text generation.
- `sentence-transformers/all-MiniLM-L6-v2` for semantic retrieval.

Expected directory layout:

```text
models/
├── Phi-4-mini-instruct-Q4_K_M.gguf
└── all-MiniLM-L6-v2/
```

Download the GGUF model from its verified source:

<https://huggingface.co/microsoft/Phi-4-mini-instruct>

The embedding model can be downloaded with the provided script.
Run it from the project root in a Python environment with
`sentence-transformers` installed:

```bash
python3 download-emb-model.py
```

This download requires internet access. The analyst loads the saved
embedding model locally at runtime.

The models used in this beta were selected to fit the available
hardware. Phi-4-mini with Q4_K_M quantization and the compact MiniLM
embedding model keep resource requirements manageable during
development. These choices are a starting point; future versions
will evaluate larger models as upgraded hardware becomes available.

## Configuration

Create your local configuration:

```bash
cp .env.example .env
```

Edit `.env` for your model and hardware. Use `.env.example` as the
complete reference for supported settings.

Main settings include:

| Setting | Purpose |
|---|---|
| `LLAMA_IMAGE` | llama.cpp server image |
| `MODEL_FILE` | GGUF filename under `models/` |
| `MODEL_NAME` | Model identifier sent by the client |
| `LLM_CONTEXT_SIZE` | Configured context capacity |
| `LLM_MAX_OUTPUT_TOKENS` | Maximum response token allowance |
| `LLM_CONTEXT_SAFETY_MARGIN` | Additional reserved context space |
| `SKILLS_DIR` | Skills repository path inside the analyst container |
| `MINILM_MODEL_PATH` | Embedding model path inside the analyst container |
| `ROUTER_TOP_K` | Maximum number of selected skills |
| `ROUTER_SEMANTIC_WEIGHT` | Semantic contribution to the combined score |
| `ROUTER_NAME_BOOST` | Lexical boost for skill-name matches |

The input budget is calculated as:

```text
input budget = context size - maximum output tokens - safety margin
```

Container paths refer to locations inside Docker, for example:

```dotenv
SKILLS_DIR=/app/skills
MINILM_MODEL_PATH=/app/models/all-MiniLM-L6-v2
```

`.env` is local configuration and is excluded from Git.
`.env.example` contains shareable example values.

## Run

From the project root:

```bash
docker compose up --build -d
```

Open the browser UI:

http://localhost:5000

View logs:

```bash
docker compose logs -f analyst llama
```

Stop the services:

```bash
docker compose down
```

The analyst UI and llama.cpp ports are bound to localhost in the
current Compose configuration.

Application code, skills, and the embedding model are copied into
the analyst image. Rebuild after changing those files.

## API

The analyst exposes:

- `POST /v1/chat/completions` — submit a prompt.
- `POST /v1/chat/selection` — remove selected skills from an
  oversized pending request.

Example:

```bash
curl http://localhost:5000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain how to investigate suspicious Windows logins."}]}'
```

Responses contain either an assistant answer and usage statistics,
or a skill-selection request when the input exceeds the budget.

## Current limitations

- Each prompt is processed independently; conversation history is
  not included.
- Selected skills supply instructions. Repository scripts are not
  automatically executed.
- Linked reference files are not automatically loaded.
- The HTTP server processes requests synchronously.
- Pending skill selections are stored in memory and disappear
  when the analyst restarts.
- Token-budget enforcement requires a compatible llama.cpp
  input-token-count endpoint.

## Skills catalog

The cybersecurity skills catalog is included as a Git submodule:

https://github.com/mukul975/Anthropic-Cybersecurity-Skills

The parent repository records a specific skills commit for
reproducible checkouts. Uncommitted edits inside the submodule
are not included in the parent project's commits.

## Attribution and licenses

This project uses:

- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [uv](https://github.com/astral-sh/uv)
- [Anthropic Cybersecurity Skills](https://github.com/mukul975/Anthropic-Cybersecurity-Skills)

Consult each upstream project's license and model terms.