import re

def strip_formatting(text: str) -> str:
    """
    Remove any basic Markdown-style formatting (bold/italic markers, etc.)
    and return only the unformatted word content.
    """
    text = re.sub(r'(\*|_){1,2}(.*?)\1{1,2}', r'\2', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text, flags=re.DOTALL)
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if __name__ == "__main__":
    sample_texts = [
        "**Hello** _world_!",
        "Some `inline code` and *italic text*",
        "Remove weird {stuff}? Maybe. **Yes**!"
    ]

    for idx, t in enumerate(sample_texts, start=1):
        cleaned = strip_formatting(t)
        print(f"Original #{idx}: {t}")
        print(f"Cleaned  #{idx}: {cleaned}\n")
