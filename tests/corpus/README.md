# Test Corpus for Simultaneous Interpretation

This directory contains the test audio corpus for evaluating interpretation pipelines.

## Directory Structure

```
corpus/
├── audio/
│   ├── short/           # Short utterances (2-5s each, ~20 files)
│   ├── medium/          # Medium segments (15-30s each, ~10 files)
│   ├── long/            # Long passages (2-5min each, ~5 files)
│   ├── technical/       # Technical/specialized content (~5 files)
│   └── conversational/  # Natural dialogue with hesitations (~5 files)
├── transcripts/
│   └── ground_truth/    # Chinese transcriptions (parallel to audio)
└── translations/
    └── reference/       # English reference translations
```

## Audio Specifications

- **Format:** WAV, 16kHz, mono, 16-bit PCM
- **Quality:** Clean recordings with minimal background noise
- **Speakers:** Multiple speakers (male/female, various accents)
- **Language:** Mandarin Chinese (普通话)

## File Naming Convention

Audio files should be named with a descriptive prefix and index:
- `short_001.wav`, `short_002.wav`, ...
- `medium_001.wav`, `medium_002.wav`, ...
- etc.

Ground truth transcriptions should match audio filenames:
- `short_001.txt` (for `short_001.wav`)

Reference translations should also match:
- `short_001.txt` (for `short_001.wav`)

## Corpus Categories

### Short Utterances (2-5 seconds)
Single sentences, greetings, simple questions. Tests basic recognition
and quick translation response.

Example content:
- 你好，很高兴见到你。
- 请问这个多少钱？
- 今天天气真好。

### Medium Segments (15-30 seconds)
Paragraph-length speech with complete thoughts. Tests handling of
longer context and more complex grammar.

### Long Passages (2-5 minutes)
Continuous speech like presentations or lectures. Tests sustained
operation and context maintenance.

### Technical Content (1-2 minutes)
Business or specialized vocabulary. Tests terminology handling and
domain adaptation.

Example domains:
- Business/finance
- Technology/software
- Medical/health
- Legal/contracts

### Conversational (2-3 minutes)
Natural dialogue with hesitations, corrections, and informal speech.
Tests real-world robustness.

Features:
- 嗯, 那个, 就是说... (filler words)
- Incomplete sentences
- Self-corrections
- Varying speech rate

## Creating Test Audio

### Option 1: Record Native Speakers
Record Chinese speakers reading prepared scripts with ground truth
transcriptions.

### Option 2: TTS Generation
Use Chinese TTS (Azure, Google, or local models) to generate
test audio from text scripts.

### Option 3: Public Datasets
Sources for Chinese speech data:
- AISHELL-1: ~170 hours of Mandarin speech
- Common Voice: Community-contributed recordings
- ST-CMDS: Short Chinese speech commands

## Reference Translations

Reference translations should be:
1. **Professional quality:** Use professional translators or
   verify GPT-4 translations manually
2. **Natural English:** Idiomatic, not word-for-word
3. **Consistent:** Same translator/style throughout category

## Quality Checklist

Before using audio files, verify:
- [ ] Audio is 16kHz mono WAV format
- [ ] No significant background noise
- [ ] Speech is clearly audible
- [ ] Ground truth transcription is accurate
- [ ] Reference translation is natural and accurate
- [ ] File naming follows convention

## Sample Data

A minimal test set is included in `samples/` with:
- 3 short utterances
- 1 medium segment
- Ground truth and reference translations

This allows quick testing before full corpus creation.
