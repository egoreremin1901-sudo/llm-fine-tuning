GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

INFERENCE_CONFIG = {
    "max_seq_length": 1024,
    "max_new_tokens": 128,
    "test_slice": (11000, 11100),
}

MODEL_CONFIG = {
    "model_name": MODEL_PATH,
    "max_seq_length": INFERENCE_CONFIG["max_seq_length"],
    "dtype": None,
    "load_in_4bit": True,
}