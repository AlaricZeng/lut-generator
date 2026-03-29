from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import io
import json
import os
import uuid
import base64
from pathlib import Path
from flask import Flask, request, jsonify, Response, stream_with_context, render_template
import numpy as np
from PIL import Image

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Server-side store keyed by session_id
lut_store: dict = {}

LUT_SIZE = 65
MAX_ITER = 6
TARGET_SCORE = 0.97


# ── AI client config ──────────────────────────────────────────────────────────

def _get_ai_config() -> tuple[str, str, str]:
    """Returns (provider, api_key, model)."""
    provider = os.environ.get("AI_PROVIDER", "gemini").lower().strip()
    api_key  = os.environ.get("AI_API_KEY", "").strip()
    model    = os.environ.get("AI_MODEL", "").strip()

    # Fallback: legacy ~/.gemini/settings.json support
    if provider == "gemini" and not api_key:
        p = Path.home() / ".gemini" / "settings.json"
        if p.exists():
            try:
                data = json.loads(p.read_text())
                api_key = data.get("apiKey") or data.get("api_key") or ""
            except Exception:
                pass

    defaults = {
        "gemini":    "gemini-2.5-pro",
        "openai":    "gpt-4o",
        "anthropic": "claude-3-5-sonnet-20241022",
    }
    if not model:
        model = defaults.get(provider, "")

    return provider, api_key, model


# ── LAB colorspace ────────────────────────────────────────────────────────────

def _srgb_to_lin(c: np.ndarray) -> np.ndarray:
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)


def _lin_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c > 0.0031308, 1.055 * np.power(np.maximum(c, 0), 1.0 / 2.4) - 0.055, 12.92 * c)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """(N,3) float64 [0,1] RGB → LAB."""
    lin = _srgb_to_lin(np.clip(rgb, 0, 1))
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float64)
    xyz = lin @ M.T
    xyz[:, 0] /= 0.95047
    xyz[:, 2] /= 1.08883
    eps = 0.008856
    f = np.where(xyz > eps, np.cbrt(np.maximum(xyz, 0)), (xyz * 903.3 + 16.0) / 116.0)
    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """(N,3) LAB → RGB [0,1], clamped."""
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    eps, kappa = 0.008856, 903.3
    xr = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * eps, fy ** 3, L / kappa)
    zr = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / kappa)
    xyz = np.stack([xr * 0.95047, yr, zr * 1.08883], axis=1)
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=np.float64)
    lin = xyz @ M_inv.T
    return np.clip(_lin_to_srgb(lin), 0.0, 1.0)


# ── LUT core ──────────────────────────────────────────────────────────────────

def generate_lut(src_img: Image.Image, tgt_img: Image.Image, corrections: dict | None = None) -> str:
    """Generate a 65³ .cube LUT that maps target colours → source style.
    Applying this LUT to the target image makes it look like the source."""
    src = np.array(src_img.convert("RGB")).reshape(-1, 3).astype(np.float64) / 255.0
    tgt = np.array(tgt_img.convert("RGB")).reshape(-1, 3).astype(np.float64) / 255.0

    MAX_PX = 60_000
    if len(src) > MAX_PX:
        src = src[np.random.choice(len(src), MAX_PX, replace=False)]
    if len(tgt) > MAX_PX:
        tgt = tgt[np.random.choice(len(tgt), MAX_PX, replace=False)]

    sl = rgb_to_lab(src)
    tl = rgb_to_lab(tgt)
    sm, ss = sl.mean(0), sl.std(0) + 1e-6  # source LAB stats
    tm, ts = tl.mean(0), tl.std(0) + 1e-6  # target LAB stats

    # Build 65³ grid — R inner, B outer (standard .cube ordering)
    lin = np.linspace(0.0, 1.0, LUT_SIZE)
    rr, gg, bb = np.meshgrid(lin, lin, lin, indexing="ij")
    grid = np.stack([rr.ravel(), gg.ravel(), bb.ravel()], axis=1)  # (N,3)

    # Transform: target colour space → source colour space
    # (grid_lab - tm) / ts normalises target, * ss + sm maps to source
    grid_lab = rgb_to_lab(grid)
    out_lab = (grid_lab - tm) / ts * ss + sm
    out = lab_to_rgb(out_lab)  # (N,3)

    # Apply corrections on top of LAB transfer
    if corrections:
        r_b = float(corrections.get("r_bias", 0.0))
        g_b = float(corrections.get("g_bias", 0.0))
        b_b = float(corrections.get("b_bias", 0.0))
        sat = float(corrections.get("saturation", 1.0))
        con = float(corrections.get("contrast", 1.0))

        out[:, 0] = np.clip(out[:, 0] + r_b, 0, 1)
        out[:, 1] = np.clip(out[:, 1] + g_b, 0, 1)
        out[:, 2] = np.clip(out[:, 2] + b_b, 0, 1)

        if sat != 1.0:
            gray = (0.299 * out[:, 0] + 0.587 * out[:, 1] + 0.114 * out[:, 2])[:, None]
            out = np.clip(gray + sat * (out - gray), 0, 1)

        if con != 1.0:
            out = np.clip(0.5 + con * (out - 0.5), 0, 1)

        for i, ch in enumerate("rgb"):
            sh = float(corrections.get(f"shadows_{ch}", 0.0))
            hi = float(corrections.get(f"highlights_{ch}", 0.0))
            lum = out[:, i]
            out[:, i] = np.clip(lum + sh * (1 - lum) + hi * lum, 0, 1)

    # Reshape to (R,G,B,3) — iterate B outer, G mid, R inner for .cube
    vol = out.reshape(LUT_SIZE, LUT_SIZE, LUT_SIZE, 3)
    lines = [f'TITLE "Generated LUT"', f"LUT_3D_SIZE {LUT_SIZE}", ""]
    for bi in range(LUT_SIZE):
        for gi in range(LUT_SIZE):
            for ri in range(LUT_SIZE):
                rv, gv, bv = vol[ri, gi, bi]
                lines.append(f"{rv:.6f} {gv:.6f} {bv:.6f}")
    return "\n".join(lines)


def _apply_corrections_to_cube(cube_text: str, corrections: dict) -> str:
    """Apply small incremental corrections directly to an existing .cube LUT's grid values.
    Works on the previous LUT rather than rebuilding from scratch — keeps changes smooth."""
    r_b  = float(corrections.get("r_bias", 0.0))
    g_b  = float(corrections.get("g_bias", 0.0))
    b_b  = float(corrections.get("b_bias", 0.0))
    sat  = float(corrections.get("saturation", 1.0))
    con  = float(corrections.get("contrast", 1.0))

    header_lines = []
    data_lines = []
    in_data = False
    lut_size = None

    for line in cube_text.splitlines():
        s = line.strip()
        if s.startswith("LUT_3D_SIZE"):
            lut_size = int(s.split()[-1])
            header_lines.append(line)
        elif not in_data and len(s.split()) == 3:
            try:
                float(s.split()[0])
                in_data = True
                data_lines.append(s)
            except ValueError:
                header_lines.append(line)
        elif in_data:
            data_lines.append(s)
        else:
            header_lines.append(line)

    out_vals = []
    for row in data_lines:
        parts = row.split()
        if len(parts) != 3:
            continue
        rv, gv, bv = float(parts[0]), float(parts[1]), float(parts[2])

        # Bias
        rv = np.clip(rv + r_b, 0, 1)
        gv = np.clip(gv + g_b, 0, 1)
        bv = np.clip(bv + b_b, 0, 1)

        # Saturation
        if sat != 1.0:
            lum = 0.299 * rv + 0.587 * gv + 0.114 * bv
            rv = float(np.clip(lum + sat * (rv - lum), 0, 1))
            gv = float(np.clip(lum + sat * (gv - lum), 0, 1))
            bv = float(np.clip(lum + sat * (bv - lum), 0, 1))

        # Contrast
        if con != 1.0:
            rv = float(np.clip(0.5 + con * (rv - 0.5), 0, 1))
            gv = float(np.clip(0.5 + con * (gv - 0.5), 0, 1))
            bv = float(np.clip(0.5 + con * (bv - 0.5), 0, 1))

        # Shadows / highlights per channel
        for val, ch in zip([rv, gv, bv], ["r", "g", "b"]):
            sh = float(corrections.get(f"shadows_{ch}", 0.0))
            hi = float(corrections.get(f"highlights_{ch}", 0.0))
            val = float(np.clip(val + sh * (1 - val) + hi * val, 0, 1))
            if ch == "r": rv = val
            elif ch == "g": gv = val
            else: bv = val

        out_vals.append(f"{rv:.6f} {gv:.6f} {bv:.6f}")

    return "\n".join(header_lines) + "\n" + "\n".join(out_vals)


def apply_lut(img: Image.Image, cube_text: str) -> Image.Image:
    """Apply a .cube LUT to an image using trilinear interpolation."""
    size = None
    data: list = []
    for line in cube_text.splitlines():
        s = line.strip()
        if s.startswith("LUT_3D_SIZE"):
            size = int(s.split()[-1])
        elif s and not s[0].isalpha() and not s.startswith("#"):
            parts = s.split()
            if len(parts) == 3:
                try:
                    data.append([float(x) for x in parts])
                except ValueError:
                    pass
    if size is None or not data:
        return img

    lut = np.array(data, dtype=np.float32)      # (size³, 3)
    src = np.array(img.convert("RGB"), dtype=np.float32) / 255.0  # (H,W,3)

    sc = src * (size - 1)
    r0 = np.clip(sc[:, :, 0].astype(int), 0, size - 2)
    g0 = np.clip(sc[:, :, 1].astype(int), 0, size - 2)
    b0 = np.clip(sc[:, :, 2].astype(int), 0, size - 2)
    r1, g1, b1 = r0 + 1, g0 + 1, b0 + 1

    rf = (sc[:, :, 0] - r0)[:, :, None]
    gf = (sc[:, :, 1] - g0)[:, :, None]
    bf = (sc[:, :, 2] - b0)[:, :, None]

    def L(r, g, b):
        return lut[r + g * size + b * size * size]

    res = (
        L(r0, g0, b0) * (1 - rf) * (1 - gf) * (1 - bf)
        + L(r1, g0, b0) * rf * (1 - gf) * (1 - bf)
        + L(r0, g1, b0) * (1 - rf) * gf * (1 - bf)
        + L(r1, g1, b0) * rf * gf * (1 - bf)
        + L(r0, g0, b1) * (1 - rf) * (1 - gf) * bf
        + L(r1, g0, b1) * rf * (1 - gf) * bf
        + L(r0, g1, b1) * (1 - rf) * gf * bf
        + L(r1, g1, b1) * rf * gf * bf
    )
    return Image.fromarray(np.clip(res * 255, 0, 255).astype(np.uint8))


# ── Image helpers ─────────────────────────────────────────────────────────────

def _thumb(img: Image.Image, max_px: int) -> Image.Image:
    c = img.copy()
    c.thumbnail((max_px, max_px), Image.LANCZOS)
    return c


def to_b64_jpeg(img: Image.Image, max_px: int = 512, quality: int = 85) -> str:
    buf = io.BytesIO()
    _thumb(img, max_px).convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def to_jpeg_bytes(img: Image.Image, max_px: int = 768) -> bytes:
    buf = io.BytesIO()
    _thumb(img, max_px).convert("RGB").save(buf, "JPEG", quality=88)
    return buf.getvalue()


def _parse_gemini_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        s, e = text.find("{"), text.rfind("}") + 1
        text = text[s:e]
    return json.loads(text)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    if "source" not in request.files or "target" not in request.files:
        return jsonify({"error": "Missing source or target image"}), 400

    try:
        src_bytes = request.files["source"].read()
        tgt_bytes = request.files["target"].read()
        src_img = Image.open(io.BytesIO(src_bytes)).convert("RGB")
        tgt_img = Image.open(io.BytesIO(tgt_bytes)).convert("RGB")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    cube = generate_lut(src_img, tgt_img)
    applied = apply_lut(tgt_img, cube)

    description = "N/A"
    provider, api_key, model = _get_ai_config()
    if provider != "none" and api_key:
        try:
            _prompt_desc = "Describe the color grading style of this image in exactly one sentence. Focus on mood, tone, and dominant colours."
            tgt_bytes_ai = to_jpeg_bytes(tgt_img)
            if provider == "gemini":
                from google import genai as _genai
                from google.genai import types as _gtypes
                _client = _genai.Client(api_key=api_key)
                resp = _client.models.generate_content(
                    model=model,
                    contents=[
                        _gtypes.Part.from_bytes(data=tgt_bytes_ai, mime_type="image/jpeg"),
                        _prompt_desc,
                    ],
                )
                description = resp.text.strip()
            elif provider == "openai":
                import openai as _openai
                _b64 = base64.b64encode(tgt_bytes_ai).decode()
                _client = _openai.OpenAI(api_key=api_key)
                resp = _client.chat.completions.create(
                    model=model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64}"}},
                        {"type": "text", "text": _prompt_desc},
                    ]}],
                )
                description = resp.choices[0].message.content.strip()
            elif provider == "anthropic":
                import anthropic as _anthropic
                _b64 = base64.b64encode(tgt_bytes_ai).decode()
                _client = _anthropic.Anthropic(api_key=api_key)
                resp = _client.messages.create(
                    model=model, max_tokens=100,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64}},
                        {"type": "text", "text": _prompt_desc},
                    ]}],
                )
                description = resp.content[0].text.strip()
        except Exception as _e:
            print(f"AI description failed ({provider}): {_e}")

    session_id = str(uuid.uuid4())
    lut_store[session_id] = {"src_bytes": src_bytes, "tgt_bytes": tgt_bytes, "initial_cube": cube}

    return jsonify({
        "cube_b64": base64.b64encode(cube.encode()).decode(),
        "preview_b64": to_b64_jpeg(applied),
        "description": description,
        "session_id": session_id,
    })


@app.route("/api/refine/stream", methods=["POST"])
def api_refine_stream():
    session_id = request.form.get("session_id", "")
    store = lut_store.get(session_id)
    if not store:
        def _err():
            yield f"event: error\ndata: {json.dumps({'message': 'Session not found'})}\n\n"
        return Response(stream_with_context(_err()), content_type="text/event-stream")

    src_img = Image.open(io.BytesIO(store["src_bytes"])).convert("RGB")
    tgt_img = Image.open(io.BytesIO(store["tgt_bytes"])).convert("RGB")

    def stream():
        provider, api_key, model = _get_ai_config()
        if provider == "none" or not api_key:
            yield f"event: error\ndata: {json.dumps({'message': 'AI not configured. Set AI_PROVIDER and AI_API_KEY in your .env file to enable iterative refinement.'})}\n\n"
            return

        def _call_ai(src_jpeg: bytes, out_jpeg: bytes, prompt: str) -> dict:
            if provider == "gemini":
                from google import genai as _genai
                from google.genai import types as _gtypes
                _client = _genai.Client(api_key=api_key)
                resp = _client.models.generate_content(
                    model=model,
                    contents=[
                        _gtypes.Part.from_bytes(data=src_jpeg, mime_type="image/jpeg"),
                        _gtypes.Part.from_bytes(data=out_jpeg, mime_type="image/jpeg"),
                        prompt,
                    ],
                )
                return _parse_gemini_json(resp.text)
            elif provider == "openai":
                import openai as _openai
                _client = _openai.OpenAI(api_key=api_key)
                b64_src = base64.b64encode(src_jpeg).decode()
                b64_out = base64.b64encode(out_jpeg).decode()
                resp = _client.chat.completions.create(
                    model=model, max_tokens=300,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_src}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_out}"}},
                        {"type": "text", "text": prompt},
                    ]}],
                )
                return _parse_gemini_json(resp.choices[0].message.content)
            elif provider == "anthropic":
                import anthropic as _anthropic
                _client = _anthropic.Anthropic(api_key=api_key)
                b64_src = base64.b64encode(src_jpeg).decode()
                b64_out = base64.b64encode(out_jpeg).decode()
                resp = _client.messages.create(
                    model=model, max_tokens=300,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_src}},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64_out}},
                        {"type": "text", "text": prompt},
                    ]}],
                )
                return _parse_gemini_json(resp.content[0].text)
            else:
                raise ValueError(f"Unknown provider: {provider}")

        # Start from V1 LUT (same one shown in Step 2 result)
        initial_cube = store.get("initial_cube") or generate_lut(src_img, tgt_img)
        current_cube_text = initial_cube

        src_jpeg = to_jpeg_bytes(src_img)

        best_score = 0.0
        best_cube_b64 = ""
        best_cube_text = current_cube_text

        LR_INIT = 0.4
        LR_DECAY = 0.85

        PROMPT = (
            "You are a professional colorist.\n"
            "Image 1: Source image (the color reference — this is the look we want to match)\n"
            "Image 2: Current output (photo after LUT was applied)\n\n"
            "Compare Image 2 to Image 1. Score how well the color style, vibe, and mood match: 0.0-1.0 (1.0 = perfect).\n"
            "Give precise corrections to close the REMAINING gap only — do not overcorrect.\n"
            "Keep corrections small and targeted.\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"score":0.75,"r_bias":0.02,"g_bias":-0.01,"b_bias":0.03,'
            '"saturation":1.05,"contrast":0.98,'
            '"shadows_r":0.01,"shadows_g":0.0,"shadows_b":-0.02,'
            '"highlights_r":-0.01,"highlights_g":0.02,"highlights_b":0.01}\n\n'
            "Constraints: bias in [-0.12, 0.12], saturation/contrast in [0.85, 1.15]."
        )

        for iteration in range(1, MAX_ITER + 1):
            lr = LR_INIT * (LR_DECAY ** (iteration - 1))

            current_output = apply_lut(tgt_img, best_cube_text)
            current_output_jpeg = to_jpeg_bytes(current_output)

            try:
                corrections = _call_ai(src_jpeg, current_output_jpeg, PROMPT)
                score = max(0.0, min(1.0, float(corrections.get("score", 0.0))))
                print(f"[refine] iter={iteration} score={score:.3f} lr={lr:.3f}", flush=True)
            except Exception as exc:
                yield f"event: error\ndata: {json.dumps({'message': str(exc), 'iteration': iteration})}\n\n"
                break

            damped = {
                "r_bias":       float(corrections.get("r_bias", 0.0)) * lr,
                "g_bias":       float(corrections.get("g_bias", 0.0)) * lr,
                "b_bias":       float(corrections.get("b_bias", 0.0)) * lr,
                "saturation":   1.0 + (float(corrections.get("saturation", 1.0)) - 1.0) * lr,
                "contrast":     1.0 + (float(corrections.get("contrast", 1.0)) - 1.0) * lr,
                "shadows_r":    float(corrections.get("shadows_r", 0.0)) * lr,
                "shadows_g":    float(corrections.get("shadows_g", 0.0)) * lr,
                "shadows_b":    float(corrections.get("shadows_b", 0.0)) * lr,
                "highlights_r": float(corrections.get("highlights_r", 0.0)) * lr,
                "highlights_g": float(corrections.get("highlights_g", 0.0)) * lr,
                "highlights_b": float(corrections.get("highlights_b", 0.0)) * lr,
            }

            current_cube_text = _apply_corrections_to_cube(best_cube_text, damped)

            new_output = apply_lut(tgt_img, current_cube_text)
            cube_b64 = base64.b64encode(current_cube_text.encode()).decode()
            preview_b64 = to_b64_jpeg(new_output)

            if score > best_score:
                best_score = score
                best_cube_b64 = cube_b64
                best_cube_text = current_cube_text

            payload = json.dumps({
                "iteration": iteration,
                "score": score,
                "cube_b64": cube_b64,
                "preview_b64": preview_b64,
            })
            yield f"event: iteration\ndata: {payload}\n\n"

            if score >= TARGET_SCORE:
                break

        done_payload = json.dumps({"best_score": best_score, "cube_b64": best_cube_b64})
        yield f"event: done\ndata: {done_payload}\n\n"

    return Response(
        stream_with_context(stream()),
        content_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(port=port, debug=False, threaded=True)
