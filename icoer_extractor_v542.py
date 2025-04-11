# icoer_extractor_v542.py (v5.4.2)

from sentence_transformers import SentenceTransformer, util
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import numpy as np
import re

# Load sentiment analyzer and transformer models
analyzer = SentimentIntensityAnalyzer()
models = {
    "light": SentenceTransformer("all-MiniLM-L6-v2"),
    "multi": SentenceTransformer("sentence-transformers/LaBSE")
}
spacy_models = {
    "en": spacy.load("en_core_web_sm"),
    "xx": spacy.load("xx_ent_wiki_sm")
}
_context_cache = {}
VALID_PROFILES = ["default", "academic", "oral-traditional", "nonlinear"]

def encode(text, model_type="light"):
    try:
        return models[model_type].encode(text, convert_to_tensor=True)
    except Exception:
        return None

def extract_semantic_similarity(text, context="The universe is coherent and structured.", model_type="light"):
    key = (context, model_type)
    if key not in _context_cache:
        _context_cache[key] = encode(context, model_type)
    emb1 = encode(text, model_type)
    emb2 = _context_cache.get(key)
    if emb1 is None or emb2 is None:
        return 0.0
    return float(util.pytorch_cos_sim(emb1, emb2)[0])

def extract_lexical_score(text, style_profile="default"):
    words = text.split()
    if not words:
        return 0.0
    unique_words = set(words)
    diversity = len(unique_words) / len(words)
    avg_length = np.mean([len(w) for w in words])
    score = 0.5 * diversity + 0.5 * (1 - abs(avg_length - 5) / 5)
    if style_profile == "nonlinear":
        score *= 1.1
    return min(max(score, 0), 1)

def extract_structural_integrity(text, lang="en", style_profile="default"):
    if style_profile not in VALID_PROFILES:
        style_profile = "default"
    try:
        nlp = spacy_models.get(lang, spacy_models["xx"])
        doc = nlp(text)
        sentence_lengths = [len(sent.text.split()) for sent in doc.sents]
        if not sentence_lengths:
            return 0.0
        variance = np.var(sentence_lengths)
        if style_profile == "oral-traditional":
            variance *= 0.75
        elif style_profile == "nonlinear":
            variance *= 0.5
        elif style_profile == "academic":
            variance *= 1.0
        score = 1 / (1 + variance)
        return min(score, 1.0)
    except Exception:
        return 0.0

def extract_logical_coherence(text, lang="en", style_profile="default"):
    try:
        nlp = spacy_models.get(lang, spacy_models["xx"])
        doc = nlp(text)
        concepts = [(ent.text, token.lemma_) for ent in doc.ents for token in ent.root.subtree]
        score = len(concepts) / max(len(doc), 1)
        if style_profile == "academic":
            score *= 1.05
        return min(score, 1.0)
    except Exception:
        return 0.0

def extract_memory_score(text, memory=[], model_type="light"):
    if not memory:
        return 0.0
    emb_text = encode(text, model_type)
    memory_scores = []
    for mem in memory:
        emb_mem = encode(mem, model_type)
        if emb_text is not None and emb_mem is not None:
            sim = float(util.pytorch_cos_sim(emb_text, emb_mem)[0])
            memory_scores.append(sim)
    if not memory_scores:
        return 0.0
    return min(np.mean(memory_scores), 1.0)

def extract_affective_tone(text):
    # Note: still uses VADER (EN only)
    try:
        vs = analyzer.polarity_scores(text)
        tone_score = abs(vs['compound'])
        return min(tone_score, 1.0)
    except Exception:
        return 0.0

def extract_all(text, memory_context=[], model_type="light", lang="en", style_profile="default"):
    return {
        "S": round(extract_semantic_similarity(text, model_type=model_type), 3),
        "L": round(extract_lexical_score(text, style_profile), 3),
        "E": round(extract_structural_integrity(text, lang, style_profile), 3),
        "C": round(extract_logical_coherence(text, lang, style_profile), 3),
        "M": round(extract_memory_score(text, memory_context, model_type), 3),
        "A": round(extract_affective_tone(text), 3)
    }
