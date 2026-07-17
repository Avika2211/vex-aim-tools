#!/usr/bin/env python3
"""Standalone ElevenLabs API key tester.

Checks whether the ELEVENLABS_API_KEY currently in your environment / .env can
actually be used to synthesize speech, and prints a clear diagnosis (auth,
plan/tier, voice permission, etc.).

Zero third-party dependencies: uses only the Python standard library, so it
runs regardless of whether `requests` or the `elevenlabs` SDK are installed.

USAGE
    python experiments/elevenlabs_keytest.py
    python experiments/elevenlabs_keytest.py --model eleven_v3
    python experiments/elevenlabs_keytest.py --model all           # try every model below
    python experiments/elevenlabs_keytest.py --voice <voice_id> --text "Hello world"

The API key is read from (in order): --api-key arg, the environment, then a
local .env file in the project root. The key is never printed in full.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# MODEL SELECTION  --  pick which model to test here (or pass --model on the CLI)
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = [
    "eleven_v3",                # newest, most expressive (may need higher tier)
    "eleven_multilingual_v2",   # stable, widely available, good quality
    "eleven_turbo_v2_5",        # low latency, good for real-time
    "eleven_flash_v2_5",        # fastest / cheapest
]

# Default model used when --model is not given. Change this line to switch.
MODEL_ID = "eleven_multilingual_v2"

# Default voice + text (override on the CLI with --voice / --text).
DEFAULT_VOICE_ID = "yowh82B72eMNrxcxHgBh"
DEFAULT_TEXT = "Hello, this is a test of the ElevenLabs API key."
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"

API_BASE = "https://api.elevenlabs.io/v1"


def find_env_file():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(here), ".env"),  # project root (parent of experiments/)
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_api_key(cli_key=None):
    if cli_key:
        return cli_key
    if os.getenv("ELEVENLABS_API_KEY"):
        return os.getenv("ELEVENLABS_API_KEY")
    env_path = find_env_file()
    if env_path:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if line.startswith("ELEVENLABS_API_KEY=") and "=" in line:
                    value = line.split("=", 1)[1].strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                        value = value[1:-1]
                    return value
    return None


def mask(key):
    if not key:
        return "(none)"
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]} (len={len(key)})"


def http_request(method, url, api_key, body=None):
    """Return (status_code, headers, raw_bytes). Never raises on HTTP errors."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("xi-api-key", api_key)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.getcode(), _headers_dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, _headers_dict(e.headers), e.read()
    except urllib.error.URLError as e:
        return None, {}, str(e.reason).encode("utf-8")


def _headers_dict(headers):
    # Case-insensitive: lowercase all header names so lookups are reliable.
    return {str(k).lower(): v for k, v in dict(headers).items()}


def _looks_like_audio(raw):
    # MP3 files start with an ID3 tag ("ID3") or an MPEG frame sync (0xFF Ex/Fx).
    return raw[:3] == b"ID3" or (len(raw) >= 2 and raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0)


def pretty_error(raw):
    try:
        obj = json.loads(raw.decode("utf-8"))
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return raw[:400].decode("utf-8", "replace")


def test_tts(api_key, model_id, voice_id, text, output_format, out_path):
    url = f"{API_BASE}/text-to-speech/{voice_id}"
    body = {"text": text, "model_id": model_id, "output_format": output_format}
    print(f"\n--- Synthesis test: model_id={model_id!r} voice_id={voice_id!r} ---")
    status, headers, raw = http_request("POST", url, api_key, body)
    ctype = headers.get("content-type", "")
    if status == 200 and ("audio" in ctype.lower() or _looks_like_audio(raw)):
        with open(out_path, "wb") as out:
            out.write(raw)
        print(f"  RESULT: SUCCESS  (HTTP 200, {len(raw)} bytes of audio)")
        print(f"  Saved audio to: {out_path}")
        return True
    print(f"  RESULT: FAILED  (HTTP {status}, Content-Type={ctype or 'n/a'})")
    print("  Response:")
    for line in pretty_error(raw).splitlines():
        print("    " + line)
    return False


def main():
    parser = argparse.ArgumentParser(description="Test the ElevenLabs API key.")
    parser.add_argument("--model", default=MODEL_ID,
                        help=f"model_id to test, or 'all'. Default: {MODEL_ID}. "
                             f"Options: {', '.join(AVAILABLE_MODELS)}")
    parser.add_argument("--voice", default=DEFAULT_VOICE_ID, help="voice_id to test")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="text to synthesize")
    parser.add_argument("--output-format", default=DEFAULT_OUTPUT_FORMAT)
    parser.add_argument("--api-key", default=None,
                        help="override the API key (otherwise env / .env is used)")
    args = parser.parse_args()

    api_key = load_api_key(args.api_key)
    print("=" * 64)
    print("ElevenLabs API key test")
    print("=" * 64)
    print(f"API key: {mask(api_key)}")
    if not api_key:
        print("\nNo ELEVENLABS_API_KEY found (checked --api-key, env, and .env).")
        sys.exit(2)

    # 1) Lightweight auth/permission probe (does not consume TTS quota).
    print("\n--- Auth probe: GET /v1/user/subscription ---")
    status, _, raw = http_request("GET", f"{API_BASE}/user/subscription", api_key)
    if status == 200:
        try:
            sub = json.loads(raw.decode("utf-8"))
            print(f"  Authenticated. tier={sub.get('tier')!r} "
                  f"characters used={sub.get('character_count')}/{sub.get('character_limit')}")
        except Exception:
            print("  Authenticated (could not parse subscription body).")
    else:
        print(f"  HTTP {status} (key may be valid but lack 'user_read' scope; continuing).")
        for line in pretty_error(raw).splitlines():
            print("    " + line)

    # 2) Actual synthesis test(s).
    models = AVAILABLE_MODELS if args.model == "all" else [args.model]
    any_ok = False
    for m in models:
        suffix = m if len(models) > 1 else "out"
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f"elevenlabs_keytest_{suffix}.mp3")
        if test_tts(api_key, m, args.voice, args.text, args.output_format, out_path):
            any_ok = True

    print("\n" + "=" * 64)
    if any_ok:
        print("OVERALL: key CAN synthesize with at least one model above. [OK]")
        sys.exit(0)
    else:
        print("OVERALL: key could NOT synthesize the requested voice/model. [FAIL]")
        print("Common causes: free-tier account (library voices blocked via API),")
        print("voice requires a higher tier, or the key lacks text_to_speech scope.")
        sys.exit(1)


if __name__ == "__main__":
    main()
