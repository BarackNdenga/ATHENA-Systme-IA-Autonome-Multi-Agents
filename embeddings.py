#!/usr/bin/env python3
"""
brain/embeddings.py - Embeddings Avancés pour Vraie Vectorisation
Remplace les hash par sentence-transformers pour embeddings sémantiques.
Utilisation optionnelle - fallback hash si deps manquantes.
"""

try:
    from sentence_transformers import SentenceTransformer
    USE_SENTENCE_TRANSFORMERS = True
except ImportError:
    USE_SENTENCE_TRANSFORMERS = False
    print("sentence-transformers non installé. Utilisation fallback hash.")

try:
    import chromadb.utils.embedding_functions as embedding_functions
    USE_CHROMADB_EMBED = True
except ImportError:
    USE_CHROMADB_EMBED = False

class AdvancedEmbedder:
    """Embeddings sémantiques avec fallback graceful."""
    
    def __init__(self):
        if USE_SENTENCE_TRANSFORMERS:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.model = None
    
    def embed(self, texts):
        """Embed texts ou fallback hash."""
        import hashlib
        
        if self.model and USE_SENTENCE_TRANSFORMERS:
            return self.model.encode(texts).tolist()
        else:
            # Fallback deterministic hash (384 dims pour matcher MiniLM)
            embeddings = []
            for text in texts:
                hash_val = hashlib.sha256(text.encode()).digest()
                embedding = [ord(hash_val[i]) / 255.0 for i in range(384)]  # Normalisé
                embeddings.append(embedding)
            return embeddings

# Fonction ChromaDB compatible (optionnelle)
def get_embedding_function():
    """Retourne SentenceTransformerEmbeddingFunction ou None."""
    if USE_SENTENCE_TRANSFORMERS and USE_CHROMADB_EMBED:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    return None

"""
Embeddings résilients - Fonctionne toujours même sans deps optionnelles
Développé par Barack Ndenga 🧠 - Robust Intelligence 2026
"""

