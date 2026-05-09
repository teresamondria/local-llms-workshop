# Local model investigations demo

This demo turns a folder of PDFs into page images, labels a few page images using a **local vision model** in Ollama, then builds a **tiny embedding search UI** over the generated descriptions.

The repo intentionally does **not** include:
- the PDF dataset folder
- the rendered images
- the SQLite DB
- any generated CSV exports

## Starting from zero (no assumptions)

This workshop assumes you might not have Python, a terminal habit, or Ollama yet. You need three things on your computer:

1. An open **terminal** (to type commands).
2. **Python 3.10 or newer** (runs the scripts; `pip` comes with it for installing small Python libraries).
3. **Ollama** (runs the LLM and embedding models on your machine).

Everything below is free for local use; you are not calling paid APIs in this demo.

### Open a terminal

- **macOS**: Spotlight (`Cmd + Space`), type `Terminal`, press Enter. Or: Finder → Applications → Utilities → Terminal.
- **Windows**: Start menu → search **PowerShell** or **Command Prompt**. (If you use **Windows Subsystem for Linux**, use that terminal instead and adapt paths.)
- **Linux**: Your distro’s terminal app (e.g. GNOME Terminal).

**Minimal navigation** (you will use these a lot):

- `pwd` — print “where am I?”
- `ls` — list files in the current folder
- `cd path/to/folder` — change folder (use quotes if the name has spaces)

Clone or download this repo, then `cd` into **this repository’s root folder** (the one that contains `requirements.txt` and the `.py` scripts) before running the commands below.

### Install Python (if you do not have it)

1. Check whether Python is already there:

   ```bash
   python3 --version
   ```

   If you see something like `Python 3.10.x` or higher, you are fine. If the command is not found, install Python.

   On some Windows setups the command is `py -3 --version` or `python --version` instead of `python3`. Use the same command consistently below (replace `python3` with `py -3` or `python` if that is what works).

2. **Install** (pick one):

   - **macOS / Windows / Linux (official installer):** [https://www.python.org/downloads/](https://www.python.org/downloads/) — download, run the installer, and **enable “Add Python to PATH”** (Windows) or follow the installer defaults (macOS).
   - **Windows (Microsoft Store):** search “Python 3.12” in the Store and install.

3. **Close and reopen** the terminal, then run `python3 --version` again.

`pip` is the tool that installs Python packages. It is usually available as:

```bash
python3 -m pip --version
```

(On Windows, if you use `py`, run `py -3 -m pip --version`.)

If that fails, try the “ensure pip” step from [Python’s pip documentation](https://pip.pypa.io/en/stable/installation/) or reinstall Python with the option to install pip.

### Install Ollama (if you do not have it)

1. Download and install from [https://ollama.com/](https://ollama.com/) (macOS app, Windows installer, or Linux instructions on the site).
2. **Start Ollama** (open the app or start the service). It should listen on `http://127.0.0.1:11434`. If the scripts say they cannot reach Ollama, the app is not running.

### Install Python libraries for this demo

From **this repo’s root** (same folder as `requirements.txt`):

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

**Windows (PowerShell or Command Prompt):**

```bash
python3 -m venv .venv
.venv\Scripts\activate
python3 -m pip install -r requirements.txt
```

After `activate`, your prompt often shows `(.venv)`. **Every time** you open a new terminal to work on this demo, `cd` into the repo root and run `source .venv/bin/activate` again (or `.venv\Scripts\activate` on Windows).

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

- Reads PDFs from `Dataset 12/` (under this repo root)
- Renders every page to PNGs under `Dataset 12 images/<pdf_stem>/page_0001.png`
- Writes/updates a SQLite DB: `dataset12_page_images.db` (in this repo root when you run the script from here)

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
