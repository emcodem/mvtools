"""Access to presets/balanced.json for the python tools: the measured factor-2 kernels (H luma, H chroma, V) as
{"taps": {"divisor": 16384, "luma_h": {...}, "chroma_h": {...}, "v": {...}}} (each kernel may override "divisor")."""
import json, pathlib

def load(name="balanced"):
    return json.loads((pathlib.Path(__file__).resolve().parent.parent / "presets" / f"{name}.json").read_text())

def measured():
    j = load()
    return {"taps": {"divisor": 16384, "luma_h": j["h"]["2"]["luma"], "chroma_h": j["h"]["2"]["chroma"],
                     "v": j["v"]["2"]["kernel"]}}
