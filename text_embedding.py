import torch
from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModel

class BGEEmbed:
    def __init__(self, device=None, model_name="BAAI/bge-large-en-v1.5"):
        """
        A simple text embedding class for the BAAI/bge-large-en-v1.5 model.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def embed_text(self, texts):
        """
        Accepts a string or a list of strings, and returns a (batch_size x hidden_dim) tensor.
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # CLS pooling (take the first token's hidden state)
            embeddings = model_output.last_hidden_state[:, 0]

        # L2-normalize the embeddings
        embeddings = normalize(embeddings, p=2, dim=1)
        return embeddings
    
if __name__ == "__main__":
    # Demo
    embedder = BGEEmbed()

    text1 = "caterpillar"
    text2 = "butterfly"

    emb1 = embedder.embed_text(text1)
    emb2 = embedder.embed_text(text2)

    # Dot product of two different embeddings
    dot_12 = torch.sum(emb1 * emb2).item()

    # Dot product of the same embedding
    dot_11 = torch.sum(emb1 * emb1).item()

    print(f"Dot product of '{text1}' and '{text2}': {dot_12}")
    print(f"Dot product of '{text1}' with itself: {dot_11}")