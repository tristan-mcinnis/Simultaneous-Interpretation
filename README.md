# Simultaneous-Interpretation

Simultaneous-Interpretation is a command-line toolkit for building real-time interpreting workflows. The project couples
microphone capture, rapid transcription, neural translation, and optional speech synthesis into a modular pipeline so you can
mix and match components that fit your environment.

## Highlights
- **Real-time transcription with whisper.cpp** – default backend uses the `whispercpp` Python bindings so you can run highly
  optimized Whisper models locally on CPU (perfect for macOS laptops). A `faster-whisper` fallback is available for systems that
  already rely on those weights.
- **Flexible OpenAI translation** – choose any publicly released OpenAI chat model (e.g., `gpt-4o`, `gpt-4o-mini`) via a command-line
  argument. The translation step respects conversation topic hints and remembers recent context for smoother phrasing.
- **Optional text-to-speech playback** – stream translations back through your speakers using the OpenAI text-to-speech API with your
  preferred voice and speed.
- **Domain dictionaries and logging** – import custom terminology mappings, review transcripts in the terminal, and export a tidy log
  to your Downloads folder after each session.

> **Note:** OpenAI has not released GPT-5 models. Use currently available models such as `gpt-4o` or `gpt-4o-mini` and update the
> `--model` argument whenever new publicly documented models appear.

## Prerequisites
- Python 3.10 or newer.
- [PortAudio](http://www.portaudio.com/) runtime for PyAudio. On macOS you can install it with `brew install portaudio`.
- An OpenAI API key stored in your environment (set `OPENAI_API_KEY` or create a `.env` file).
- Whisper model weights for whisper.cpp. Download a `.bin` model (for example `ggml-base.en.bin`) from the official
  whisper.cpp repository and place it somewhere accessible.

## Installation
```bash
git clone https://github.com/yourusername/simultaneous-interpretation.git
cd simultaneous-interpretation
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
pip install -r requirements.txt
```

Add the source tree to your Python path so the package is importable without installation:
```bash
export PYTHONPATH="$PWD/src"
```

## Configuration
Create an `.env` file or export environment variables before running the CLI:
```bash
OPENAI_API_KEY=sk-...
```
Optional flags:
- `--dictionary /path/to/file.txt` to inject custom term mappings.
- `--topic "Quarterly finance review"` to bias translations toward your subject.
- `--log-file ~/Documents/interpreter.log` to change where the rolling log is written.

## Usage
List available microphones and speakers:
```bash
python -m siminterp --list-devices
```

Start an interpreting session (example translates from English to French, plays back audio, and uses a custom whisper.cpp model file):
```bash
python -m siminterp \
  --input-language en \
  --target-language fr \
  --input-device 1 \
  --output-device 3 \
  --translate \
  --tts \
  --model gpt-4o \
  --tts-model gpt-4o-mini-tts \
  --voice alloy \
  --whisper-model ~/Models/ggml-base.en.bin
```
Press `CTRL+C` to stop. The application gracefully shuts down background workers and stores a timestamped transcript in your
Downloads folder.

### Choosing a transcription backend
The CLI defaults to `--transcriber whispercpp`. If you prefer the legacy `faster-whisper` workflow you can swap with:
```bash
python -m siminterp --transcriber faster-whisper --whisper-model medium
```

### CLI reference
Run `python -m siminterp --help` to view the full list of options, including:
- `--temperature` to adjust translation creativity.
- `--history` to control how many previous translations are shared as context.
- `--tts-speed` to fine-tune playback speed.
- `--phrase-time-limit` and `--ambient-duration` to tailor microphone capture windows.

## Custom dictionary format
```
term1=translation1
term2=translation2
...
```
These replacements occur before translation, allowing you to enforce company-specific nomenclature or names.

## Logging & exports
Session transcripts and translations stream to the console via Rich and are appended to `logfile.txt` (configurable). When you
exit, the combined transcript is written to your Downloads directory with separate sections for source and translated text.

## Troubleshooting tips
- Ensure the whisper.cpp model path matches the actual filename (e.g., `ggml-base.en.bin`).
- If PyAudio cannot find devices on macOS, open *System Preferences → Security & Privacy → Microphone* and grant terminal access.
- For the fastest start-up, pre-download OpenAI models you plan to use so you can quickly swap via the `--model` argument when updates arrive.
