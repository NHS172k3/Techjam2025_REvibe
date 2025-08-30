#!/usr/bin/env python3
import argparse
import math
import os
import re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

# Optional: enable zero-shot scoring (requires transformers)
_ZS_AVAILABLE = True
try:
    from transformers import pipeline
except Exception:
    _ZS_AVAILABLE = False


# ----------------------------
# 1) Simple text utilities
# ----------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")

def normalize_text(t: str) -> str:
    t = t or ""
    t = t.lower()
    t = URL_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip()
    return t

def chunk_text(t: str, max_tokens: int = 256) -> List[str]:
    """
    Rough chunker by words so long captions don't blow up the model.
    """
    words = t.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunks.append(" ".join(words[i:i + max_tokens]))
    return chunks


# ----------------------------
# 2) Lexicons (quick heuristics)
# ----------------------------
EDU_POS = {
    "tutorial", "how to", "step by step", "lecture", "explains", "explained",
    "demonstration", "exercise", "walkthrough", "course", "class", "syllabus",
    "definitions", "concepts", "example", "examples", "practice", "quiz",
    "guide", "learn", "learning", "teaches", "teaching", "workshop", "school"
    "lesson", "curriculum", "study guide", "exam prep", "revision", "homework help",
    "training", "bootcamp", "masterclass", "seminar", "presentation", "educational",
    "tutorial series", "knowledge", "instruction", "explanation", "tips and tricks",
    "faq", "reference", "whiteboard", "slides", "case study", "lab session",
    "exercise set", "coding challenge", "walk through", "mentorship", "hands-on"
}
EDU_NEG = {
    "prank", "reaction video", "compilation", "meme", "vlog", "rant",
    "asmr", "unboxing", "haul", "drama",
    "funny moments", "fails", "challenge video", "roast", "gossip",
    "spoof", "skit", "shorts", "livestream drama", "beef", "exposed",
    "giveaway", "shopping spree", "day in the life", "storytime",
    "tiktok trend", "dance challenge", "lip sync", "fashion haul",
    "celebrity news", "shipping drama"
}
ECO_POS = {
    "sustainability", "eco", "environment", "recycling", "recycle", "compost",
    "renewable", "solar", "wind", "geothermal", "hydro", "biodiversity",
    "conservation", "carbon", "emissions", "net zero", "climate", "climate change",
    "heat pump", "insulation", "energy efficiency", "green building",
    "circular economy", "lifecycle", "life cycle", "lca", "sustainable",
    "carbon footprint", "low carbon", "clean energy", "green energy",
    "zero waste", "plastic free", "plastic reduction", "ocean cleanup",
    "deforestation", "afforestation", "reforestation", "renewables",
    "electric vehicle", "ev charging", "bike sharing", "public transport",
    "urban farming", "vertical farming", "organic", "permaculture",
    "sustainable fashion", "fair trade", "eco-friendly", "water conservation",
    "solar panels", "wind turbines", "green tech", "decarbonization",
    "offsetting", "carbon credits", "renewable transition", "fossil fuel phase-out"
}
ECO_NEG = {
    "climate hoax", "anti-environment", "anti environmental",
    "drill baby", "increase emissions",
    "climate scam", "fake science", "greenwashing", "pro oil",
    "fossil fuels forever", "gas guzzler", "anti renewable",
    "drill more", "oil boom", "pro fracking", "stop wind farms",
    "ban solar", "anti climate", "fake climate crisis", "warming is natural",
    "deny global warming", "clean coal", "pro pipeline", "oil independence"
}

def keyword_score(text: str, positives: set, negatives: set) -> float:
    t = normalize_text(text)
    pos = sum(1 for p in positives if p in t)
    neg = sum(1 for n in negatives if n in t)
    raw = pos - neg
    return 1.0 / (1.0 + math.exp(-raw))

class ZeroShotScorer:
    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: int = -1):
        if not _ZS_AVAILABLE:
            raise RuntimeError("transformers is not installed. Run: pip install transformers torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu")
        self.clf = pipeline("zero-shot-classification", model=model_name, device=device)

    def entailment_score(self, chunks: List[str], hypothesis: str) -> float:
        if not chunks:
            return 0.0
        results = self.clf(chunks, candidate_labels=[hypothesis], hypothesis_template="This text {}.")
        if isinstance(results, dict):
            scores = [results["scores"][0]]
        else:
            scores = [r["scores"][0] for r in results]
        return float(np.mean(scores))

def educational_zero_shot_score(chunks: List[str], zsc: ZeroShotScorer) -> float:
    hyp = "is instructional and teaches a concept step by step for learners"
    return zsc.entailment_score(chunks, hyp)

def eco_zero_shot_score(chunks: List[str], zsc: ZeroShotScorer) -> float:
    hyp = "discusses environmentally sustainable practices with actionable guidance"
    return zsc.entailment_score(chunks, hyp)

def score_captions(
    captions: str,
    zsc: ZeroShotScorer = None
) -> Tuple[float, float, float, Dict[str, float]]:
    t = normalize_text(captions)
    chunks = chunk_text(t, max_tokens=256)
    edu_kw = keyword_score(t, EDU_POS, EDU_NEG)
    eco_kw = keyword_score(t, ECO_POS, ECO_NEG)

    if zsc is not None:
        edu_zs = educational_zero_shot_score(chunks, zsc)
        eco_zs = eco_zero_shot_score(chunks, zsc)
    else:
        edu_zs = 0.0
        eco_zs = 0.0

    EDU_W_ZS, EDU_W_KW = 0.7, 0.3
    ECO_W_ZS, ECO_W_KW = 0.7, 0.3

    educational = EDU_W_ZS * edu_zs + EDU_W_KW * edu_kw
    eco = ECO_W_ZS * eco_zs + ECO_W_KW * eco_kw
    society = float(np.mean([educational, eco]))  
    components = {
        "edu_kw": edu_kw,
        "edu_zs": edu_zs,
        "eco_kw": eco_kw,
        "eco_zs": eco_zs,
    }
    return float(educational), float(eco), society, components


def score_file(
    input_path,
    text_col="subtitles",
    nli_model="facebook/bart-large-mnli",
    device=-1
):

    if input_path.lower().endswith(".parquet"):
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)

    if text_col not in df.columns:
        raise KeyError(f"Input file must contain column '{text_col}'")

    if not _ZS_AVAILABLE:
        raise RuntimeError("transformers not installed. Run: pip install transformers torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu")
    zsc = ZeroShotScorer(model_name=nli_model, device=device)

    edus, ecos, socs, edu_kw_list, edu_zs_list, eco_kw_list, eco_zs_list = ([] for _ in range(7))
    for txt in df[text_col].fillna("").astype(str).tolist():
        edu, eco, soc, comp = score_captions(txt, zsc=zsc)
        edus.append(edu)
        ecos.append(eco)
        socs.append(soc)
        edu_kw_list.append(comp["edu_kw"])
        edu_zs_list.append(comp["edu_zs"])
        eco_kw_list.append(comp["eco_kw"])
        eco_zs_list.append(comp["eco_zs"])

    out = df.copy()
    out["score_edu_kw"] = edu_kw_list
    out["score_edu_zs"] = edu_zs_list
    out["score_eco_kw"] = eco_kw_list
    out["score_eco_zs"] = eco_zs_list
    out["score_educational"] = 0.7 * out["score_edu_zs"] + 0.3 * out["score_edu_kw"]
    out["score_eco"] = 0.7 * out["score_eco_zs"] + 0.3 * out["score_eco_kw"]
    out["total_score_raw"] = 0.5 * (out["score_educational"] + out["score_eco"])

    min_v = float(out["total_score_raw"].min())
    max_v = float(out["total_score_raw"].max())
    if max_v > min_v:
        out["total_score"] = (out["total_score_raw"] - min_v) / (max_v - min_v)
    else:
        out["total_score"] = 0.5

    out["multiplier"] = 0.9 + 0.2 * out["total_score"]

    if "score_society_contributing" in out.columns:
        out = out.drop(columns=["score_society_contributing"])    
    return out

if __name__ == "__main__":
    out = score_file("full_video_dataset_with_comments_scored_per_comment.csv","subtitles")
    print(out)
