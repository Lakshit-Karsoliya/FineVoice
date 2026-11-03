<h1 align="center">FineVoice</h1><br>

### Your voice. Refined. No subscriptions.

FineVoice is a free, open-source voice isolation and enhancement application designed to make professional-quality audio processing accessible to everyone.

Record your voice directly or upload an existing audio file. FineVoice takes care of the rest.

### How it works

**01 — Record or Upload**

Record your voice directly in FineVoice or upload an existing audio recording.

**02 — Voice Isolation**

FineVoice isolates your voice from background noise and unwanted sounds using open-source AI models.

**03 — Voice Enhancement**

The isolated audio is enhanced to improve clarity, presence, and overall vocal quality.

**04 — Post-processing & Download**

The processed voice goes through final audio post-processing. Once complete, preview and download your enhanced recording.

---

## Why FineVoice?

Most voice enhancement tools put useful features behind monthly subscriptions.

FineVoice takes a different approach.

- Completely free to use.
- Powered by open-source AI models.
- No monthly subscriptions.
- Record or upload audio.
- Voice isolation and enhancement in one workflow.
- Download your processed audio.

**Built on the idea that useful technology should be accessible to everyone.**

---

## Installation

### Requirements

- macOS
- Python 3.11
- FFmpeg

Install the required dependencies:

```bash
brew install python@3.11 ffmpeg
```

*Skip anything you already have installed.*

### Set up the environment

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/app.git
cd app
```

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Upgrade pip and install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running FineVoice

### Launch the application

```bash
streamlit run app.py --server.fileWatcherType none
```

### Run the API separately

Alternatively, launch the FastAPI server:

```bash
uvicorn engine.main:app --port 8000
```

---

## Technology

FineVoice is built using open-source technologies and AI models for voice isolation, enhancement, and audio processing.

---

## Philosophy

**No subscriptions. No unnecessary barriers. Just useful technology.**

FineVoice is built to give everyone access to better voice recordings without having to pay recurring fees.

---

## License

See [LICENSE](LICENSE) for licensing details.

---

<p align="center">
  **FineVoice**
  <br>
  <sub>Made with purpose. Powered by open source.</sub>
</p>
