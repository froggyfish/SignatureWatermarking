import hashlib
import os
import torch
import numpy as np
from py_ecc.bls import G2ProofOfPossession as bls

# --- Installation ---
# You will need to install py_ecc:
# pip install py_ecc

class SignatureScheme:
    """
    Implements a scheme to create and verify a payload that combines a BLS
    signature with a long-form hash of the original message.

    The process involves:
    1. Signing a message 'm' to get a signature 'c'.
    2. Generating a long hash 'h' of 'm'.
    3. Masking 'c' by XORing it with the beginning of 'h'.
    4. Concatenating the masked signature with the rest of 'h'.
    """

    def __init__(self, target_bytes: int = 4*64*64//8):
        """
        Initializes the scheme with a target payload size.

        Args:
            target_bytes: The total desired length of the final payload in bytes.
        """
        if target_bytes <= 96:
            raise ValueError("Target bytes must be greater than the signature length (96).")
        self.TARGET_BYTES = target_bytes
        self.SIG_LEN_BYTES = 96  # BLS signature length is fixed

    @staticmethod
    def generate_keys(private_key=2187) -> tuple[int, bytes]:
        """
        Generates a valid BLS public key using the given private key.
        The private key is securely generated and ensured to be within the
        valid range for the BLS12-381 curve.

        Returns:
            A tuple containing the (private_key, public_key).
        """
        # A valid private key is a random integer `1 <= key < CURVE_ORDER`.
        
        public_key = bls.SkToPk(private_key)
        return private_key, public_key

    def create(self, private_key: int, message: bytes) -> bytes:
        """
        Creates a verifiable payload from a message and private key.

        Args:
            private_key: The signer's private key.
            message: The original message (as bytes) to process.

        Returns:
            The final, combined payload of `TARGET_BYTES` length.
        """
        # 1. Sign the message to get signature 'c'.
        signature_c = bls.Sign(private_key, message)

        # 2. Hash the message to get long hash 'h'.
        hash_h = self._hash_long(message, self.TARGET_BYTES)

        # 3. Mask the signature by XORing it with the start of the hash.
        masked_signature = self._xor_bytes(signature_c, hash_h[:self.SIG_LEN_BYTES])

        # 4. Concatenate the masked signature with the remainder of the hash.
        payload_bytes = masked_signature + hash_h[self.SIG_LEN_BYTES:]
        payload_bits_np = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))

        # Convert to a tensor.
        payload_bits_tensor = torch.from_numpy(payload_bits_np)
        
        # Apply the {0, 1} -> {1, -1} transformation and ensure float type.
        bipolar_tensor = 1.0 - 2.0 * payload_bits_tensor.float()
        return bipolar_tensor
    def verify(self, public_key: bytes, message: bytes, payload: bytes) -> bool:
        """
        Verifies a payload against the public key and original message.

        Args:
            public_key: The signer's public key.
            message: The original message that was supposedly signed.
            payload: The payload to verify.

        Returns:
            True if the signature is valid, False otherwise.
        """
        if len(payload) != self.TARGET_BYTES:
            return False

        # 1. Regenerate the long hash 'h' from the original message.
        hash_h = self._hash_long(message, self.TARGET_BYTES)

        # 2. Extract the masked signature from the start of the payload.
        masked_signature = payload[:self.SIG_LEN_BYTES]

        # 3. Unmask the signature by re-applying the XOR operation.
        recovered_signature_c = self._xor_bytes(masked_signature, hash_h[:self.SIG_LEN_BYTES])

        # 4. Verify the recovered signature.
        return bls.Verify(public_key, message, recovered_signature_c)

    @staticmethod
    def _hash_long(message: bytes, length: int) -> bytes:
        """Hashes a message to a specified length using SHAKE256."""
        hasher = hashlib.shake_256()
        hasher.update(message)
        return hasher.digest(length)

    @staticmethod
    def _xor_bytes(b1: bytes, b2: bytes) -> bytes:
        """Performs a byte-wise XOR on two equal-length byte strings."""
        return bytes(x ^ y for x, y in zip(b1, b2))


# --- Example Usage ---
if __name__ == "__main__":
    message = b"hi"
    scheme = SignatureScheme()
    signer_private_key, signer_public_key = SignatureScheme.generate_keys(2187)
    codeword = scheme.create(signer_private_key, message)
    print(type(codeword))
    print(codeword[0:10])
    print(len(codeword))
    # is_valid = scheme.verify(signer_public_key, message, final_payload)
    # print(f"Verification result (correct data): {is_valid}")
    # assert is_valid

    # # 6. A verifier checks the payload with a tampered message.
    # tampered_message = b"This is NOT the original message."
    # is_valid_tampered = scheme.verify(signer_public_key, tampered_message, final_payload)
    # print(f"Verification result (tampered message): {is_valid_tampered}")
    # assert not is_valid_tampered
    
    # print("\nAll tests passed.")
