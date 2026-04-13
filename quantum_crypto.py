#!/usr/bin/env python3
"""
core/quantum_crypto.py - Cryptographie Post-Quantique réelle
Utilise liboqs-python (Open Quantum Safe) pour Kyber + Dilithium.
Fallback sur AES-256-GCM + ECDSA si OQS non disponible.
"""

import os
import base64
import logging
from typing import Tuple, Dict

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger(__name__)

try:
    import oqs
    OQS_AVAILABLE = True
    logger.info("liboqs disponible: cryptographie post-quantique active.")
except ImportError:
    OQS_AVAILABLE = False
    logger.warning("liboqs non disponible: fallback AES-256-GCM + Ed25519.")


class QuantumSecureChannel:
    """Canal sécurisé: Kyber1024 (KEM) + Dilithium3 (signature) si OQS dispo,
    sinon AES-256-GCM + Ed25519."""

    def __init__(self):
        if OQS_AVAILABLE:
            self._kem = oqs.KeyEncapsulation("Kyber1024")
            self._sig = oqs.Signature("Dilithium3")
            self.public_key_kem = self._kem.generate_keypair()
            self.public_key_sig = self._sig.generate_keypair()
            self._algorithm = "Kyber1024+Dilithium3"
        else:
            self._signing_key = Ed25519PrivateKey.generate()
            self._verify_key = self._signing_key.public_key()
            self._algorithm = "AES-256-GCM+Ed25519"
        logger.info(f"QuantumSecureChannel inité avec {self._algorithm}")

    def encrypt_message(self, message: str) -> Dict[str, str]:
        """Chiffre un message avec la clé symétrique + signature."""
        msg_bytes = message.encode("utf-8")

        if OQS_AVAILABLE:
            # KEM: encapsuler une clé symétrique
            ciphertext_kem, shared_secret = self._kem.encap_secret(self.public_key_kem)
            key = shared_secret[:32]
            nonce = os.urandom(12)
            aesgcm = AESGCM(key)
            ct = aesgcm.encrypt(nonce, msg_bytes, None)
            signature = self._sig.sign(ct)
            return {
                "ciphertext": base64.b64encode(ct).decode(),
                "kem_ct": base64.b64encode(ciphertext_kem).decode(),
                "nonce": base64.b64encode(nonce).decode(),
                "signature": base64.b64encode(signature).decode(),
                "algorithm": self._algorithm
            }
        else:
            nonce = os.urandom(12)
            key = os.urandom(32)
            aesgcm = AESGCM(key)
            ct = aesgcm.encrypt(nonce, msg_bytes, None)
            signature = self._signing_key.sign(ct)
            return {
                "ciphertext": base64.b64encode(ct).decode(),
                "key": base64.b64encode(key).decode(),
                "nonce": base64.b64encode(nonce).decode(),
                "signature": base64.b64encode(signature).decode(),
                "algorithm": self._algorithm
            }

    def decrypt_message(self, data: Dict[str, str]) -> str:
        """Déchiffre et vérifie la signature."""
        ct = base64.b64decode(data["ciphertext"])
        nonce = base64.b64decode(data["nonce"])
        sig = base64.b64decode(data["signature"])

        if OQS_AVAILABLE:
            ciphertext_kem = base64.b64decode(data["kem_ct"])
            shared_secret = self._kem.decap_secret(ciphertext_kem)
            key = shared_secret[:32]
            self._sig.verify(ct, sig, self.public_key_sig)
        else:
            key = base64.b64decode(data["key"])
            self._verify_key.verify(sig, ct)

        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ct, None).decode("utf-8")

    @property
    def algorithm(self) -> str:
        return self._algorithm
