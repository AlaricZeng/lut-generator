# 🎨 Color LUT Generator

Upload a **source image** (color reference) and a **target photo** — get a professional `.cube` LUT that transfers the source's color style onto your photo, refined iteratively by AI.

Compatible with DaVinci Resolve, Premiere Pro, Final Cut Pro, Lightroom, Photoshop, and any tool that accepts `.cube` LUTs.

## How It Works

1. **Upload** a source image (the color look you want) and a target photo (the one to recolor)
2. **Generate** — a 65³ LUT is built using LAB colorspace statistical transfer
3. **Refine** — AI compares the result to the source and iteratively adjusts the LUT (up to 6 rounds) until the color style matches

## Features

- 🎨 **65×65×65 3D LUT** — high-precision `.cube` output
- 🔬 **LAB colorspace** — perceptually accurate color matching (not just RGB math)
- 🤖 **AI iterative refinement** — scores each iteration and applies targeted corrections
- 📊 **Per-iteration downloads** — grab any intermediate LUT
- 🌐 **Multi-user ready** — server-side session storage supports simultaneous users

## Requirements

- Python 3.9+
- pip

## Setup

### 1. Clone

```bash
git clone https://github.com/AlaricZeng/lut-generator.git
cd lut-generator
```

### 2. Install

```bash
pip install -r requirements.txt
```

> Also install your AI provider package — see [AI Setup](#ai-setup) below.

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` with your API key.

### 4. Run

```bash
bash run.sh
# or: python3 app.py
```

Open **http://localhost:5003**

---

## AI Setup

AI is **optional** — the LAB transfer LUT works without it. AI adds iterative refinement that scores and adjusts the LUT to better match the source color style.

Set in `.env`:

```env
AI_PROVIDER=gemini    # gemini | openai | anthropic | none
AI_API_KEY=your_key
AI_MODEL=             # leave blank for defaults
```

| Provider | `AI_PROVIDER` | Default model | Install |
|---|---|---|---|
| Google Gemini | `gemini` | `gemini-2.5-pro` | `pip install google-genai` |
| OpenAI | `openai` | `gpt-4o` | `pip install openai` |
| Anthropic Claude | `anthropic` | `claude-3-5-sonnet-20241022` | `pip install anthropic` |
| Disabled | `none` | — | — |

**Get API keys:**
- Gemini: [aistudio.google.com](https://aistudio.google.com) (free tier available)
- OpenAI: [platform.openai.com](https://platform.openai.com)
- Anthropic: [console.anthropic.com](https://console.anthropic.com)

---

## Importing the .cube File

| App | Where to import |
|---|---|
| **DaVinci Resolve** | Color page → LUTs → Right-click → Import LUT |
| **Premiere Pro** | Lumetri Color → Creative → Look → Browse |
| **Final Cut Pro** | Effects → Color → Custom LUT → Choose file |
| **Lightroom** | Profile Browser → Import Profiles |
| **Photoshop** | Layer → New Adjustment Layer → Color Lookup → Load 3D LUT |

---

## Custom Port

```bash
PORT=8080 python3 app.py
```

---

## License

MIT
