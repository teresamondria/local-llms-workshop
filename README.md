# DATASET_12_DEMO (local-model workshop demo)

This demo turns a folder of PDFs into page images, labels a few page images using a **local vision model** in Ollama, then builds a **tiny embedding search UI** over the generated descriptions.

The repo intentionally does **not** include:
- the PDF dataset folder
- the rendered images
- the SQLite DB
- any generated CSV exports

## Prereqs

- **Python 3.10+**
- **Ollama** installed and running (`https://ollama.com/`)

## Install Python deps

From `DATASET_12_DEMO/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Install Ollama models

### Vision (for image labeling)

This demo uses **`llava`** by default in `label_images_ollama_vision.py`.

```bash
ollama pull llava
```

### Embeddings (for search)

The search scripts use Ollama embeddings with **`nomic-embed-text`** by default.

```bash
ollama pull nomic-embed-text
```

## What each script does

### `render_pdfs_to_images.py`

- Reads PDFs from `DATASET_12_DEMO/Dataset 12/`
- Renders every page to PNGs under `DATASET_12_DEMO/Dataset 12 images/<pdf_stem>/page_0001.png`
- Writes/updates a SQLite DB: `DATASET_12_DEMO/dataset12_page_images.db`

Run:

```bash
python3 render_pdfs_to_images.py
```

### `label_images_ollama_vision.py`

- Reads page image paths from `dataset12_page_images.db`
- Sends each PNG to Ollama `POST /api/chat` with a vision model (default **`llava`**)
- Writes:
  - `vision_category` (`photo` or `document`)
  - `vision_description` (multi-sentence description)
  - `vision_model`, timestamps, any errors

Demo run (label only 3 pages):

```bash
python3 label_images_ollama_vision.py --model llava --limit 3
```

### `embed_image_descriptions_ollama.py`

- Reads `vision_description` from the DB
- Calls Ollama `POST /api/embeddings` (default **`nomic-embed-text`**)
- Stores vectors back into SQLite (`desc_embedding_json`, plus metadata)

Demo run (embed only 3 descriptions):

```bash
python3 embed_image_descriptions_ollama.py --model nomic-embed-text --limit 3
```

### `search_image_descriptions_ui.py`

- Starts a tiny local UI (Gradio)
- Embeds your query with Ollama embeddings
- Finds the most similar **image descriptions** via cosine similarity
- Shows the matching page image(s) + description(s)

Run:

```bash
python3 search_image_descriptions_ui.py
```

## End-to-end tutorial (start to finish)

1) **Put PDFs in the expected folder**

Create this folder and drop a few PDFs in it:

```bash
mkdir -p "Dataset 12"
```

2) **Render PDFs to images + create the DB**

```bash
python3 render_pdfs_to_images.py
```

3) **Label just 3 page images (demo-friendly)**

```bash
python3 label_images_ollama_vision.py --model llava --limit 3
```

4) **Embed those 3 descriptions**

```bash
python3 embed_image_descriptions_ollama.py --model nomic-embed-text --limit 3
```

5) **Start the search UI**

```bash
python3 search_image_descriptions_ui.py
```

Then open the local URL printed in your terminal (default `http://127.0.0.1:7860`).

### Example queries

- `handwritten`
- `letterhead`
- `table of numbers`
- `fax cover sheet`
- `passport photo`

