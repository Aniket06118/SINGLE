import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel


class CLIPEmbeddings(nn.Module):
    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        device=None,
        max_length=77,
        cache_enabled=True
    ):
        super().__init__()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.cache_enabled = cache_enabled

        # Load tokenizer + model
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        #using frozen model for Stable Diff
        for p in self.model.parameters():
            p.requires_grad = False

        # Simple cache: text to embedding
        self.cache = {}

        #null embedding for CFG
        self.null_embedding = self._encode([""])

    def _tokenize(self, texts):
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    @torch.no_grad()
    def _encode(self, texts):
        tokens = self._tokenize(texts)
        input_ids = tokens.input_ids.to(self.device)

        outputs = self.model(input_ids=input_ids)
        return outputs.last_hidden_state  # (B, 77, 768) NEED TO PROJECT THIS INTO mode_dim WHEN IN UNET!!!

    def forward(self, texts):
        """
        texts: list[str]
        returns: (B, 77, 768)
        """

        # Normalize input
        if isinstance(texts, str):
            texts = [texts]

        #cache checking
        if self.cache_enabled:
            uncached = [t for t in texts if t not in self.cache]

            if len(uncached) > 0:
                new_embs = self._encode(uncached)
                for text, emb in zip(uncached, new_embs):
                    self.cache[text] = emb.unsqueeze(0)

            embeddings = torch.cat([self.cache[t] for t in texts], dim=0)
            return embeddings.to(self.device)

        #no cache path
        return self._encode(texts)

    def get_null_embedding(self, batch_size=1):
        """
        Returns null embedding for CFG
        shape: (B, 77, 768)
        """
        return self.null_embedding.repeat(batch_size, 1, 1)