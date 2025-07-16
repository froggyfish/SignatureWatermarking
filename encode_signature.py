import hashlib
import os
import torch
import numpy as np
from py_ecc.bls import G2ProofOfPossession as bls

# --- Installation ---
# You will need to install py_ecc, galois, and numpy:
# pip install py_ecc galois numpy
try:
    import galois
except ImportError:
    print("Please install the galois library: pip install galois numpy")
    exit()


class SignatureScheme:
    """
    Implements a scheme to create and verify a payload that combines a BLS
    signature with a long-form hash of the original message.
    Includes Reed-Solomon error correction to handle noisy channels.
    """

    def __init__(self, target_bytes: int = 4*64*64):
        """
        Initializes the scheme with a target payload size.

        Args:
            target_bytes: The total desired length of the final payload in bytes.
        """
        # Reed-Solomon parameters: encode the 96-byte signature into a 255-byte block.
        self.RS_N = 255  # Codeword length
        self.RS_K = 96   # Message length (must match signature length)

        if target_bytes <= self.RS_N:
            raise ValueError(f"Target bytes must be greater than the RS codeword length ({self.RS_N}).")
        
        self.TARGET_BYTES = target_bytes
        self.SIG_LEN_BYTES = 96  # BLS signature length is fixed

    @staticmethod
    def generate_keys(private_key=2187) -> tuple[int, bytes]:
        """
        Generates a valid BLS public key using the given private key.
        """
        # In a real application, a key should be securely and randomly generated.
        # This deterministic key is for demonstration purposes.
        public_key = bls.SkToPk(private_key)
        return private_key, public_key

    def create(self, private_key: int, message: bytes) -> torch.Tensor:
        """
        Creates a verifiable payload, encodes it with Reed-Solomon,
        and returns a bipolar tensor.
        """
        # 1. Sign the message to get the 96-byte signature 'c'.
        signature_c = bls.Sign(private_key, message)

        # 2. Add Reed-Solomon error correction to the signature.
        rs = galois.ReedSolomon(self.RS_N, self.RS_K)
        
        # FIX: Convert bytes to a NumPy array before encoding with Galois.
        signature_np = np.frombuffer(signature_c, dtype=np.uint8)
        c_encoded_gf = rs.encode(signature_np) # This is a Galois Field Array
        c_encoded = bytes(c_encoded_gf) # Convert back to bytes for concatenation

        # 3. Hash the message to get long hash 'h'.
        hash_h = self._hash_long(message, self.TARGET_BYTES)

        # 4. Create the payload before masking.
        #    (Encoded Signature || Rest of Hash)
        payload_before_xor = c_encoded + hash_h[self.RS_N:]
        
        # 5. Mask the entire payload with the hash.
        masked_payload = self._xor_bytes(payload_before_xor, hash_h)

        # 6. Convert the final bytes to a bipolar tensor.
        payload_bits_np = np.unpackbits(np.frombuffer(masked_payload, dtype=np.uint8))
        payload_bits_tensor = torch.from_numpy(payload_bits_np)
        bipolar_tensor = 1.0 - 2.0 * payload_bits_tensor.float()
        
        return bipolar_tensor

    def decode_and_verify(self, public_key: bytes, message: bytes, noisy_tensor: torch.Tensor) -> bool:
        """
        Decodes a noisy tensor, corrects errors using Reed-Solomon,
        and verifies the underlying signature.

        Args:
            public_key: The signer's public key.
            message: The original message that was supposedly signed.
            noisy_tensor: The received {-1, 1} tensor, potentially with errors.

        Returns:
            True if the signature is valid after error correction, False otherwise.
        """
        # 1. Convert tensor back to (potentially corrupted) bytes.
        reconstructed_bits = (1.0 - noisy_tensor.view(-1)) / 2.0
        noisy_masked_payload = np.packbits(reconstructed_bits.round().byte().numpy()).tobytes()

        if len(noisy_masked_payload) != self.TARGET_BYTES:
            return False

        # 2. Regenerate the hash 'h' to use as the unmasking key.
        hash_h = self._hash_long(message, self.TARGET_BYTES)

        # 3. Unmask the payload to reveal the (noisy) encoded signature and hash remainder.
        noisy_payload_before_xor = self._xor_bytes(noisy_masked_payload, hash_h)
        
        # 4. Extract the noisy encoded signature.
        noisy_c_encoded = noisy_payload_before_xor[:self.RS_N]

        # 5. Decode using Reed-Solomon to correct errors.
        rs = galois.ReedSolomon(self.RS_N, self.RS_K)
        try:
            # FIX: Convert the byte slice to a NumPy array before decoding.
            noisy_c_encoded_np = np.frombuffer(noisy_c_encoded, dtype=np.uint8)
            recovered_signature_gf = rs.decode(noisy_c_encoded_np)
            # FIX: Convert the decoded Galois Field array back to bytes for verification.
            recovered_signature_c = bytes(recovered_signature_gf)
        except galois.errors.ReedSolomonError:
            # This occurs if there are too many errors to correct.
            return False

        # 6. Verify the recovered, pristine signature.
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
    # Note: target_bytes must be 16384 for a 4*64*64*8 bit tensor
    scheme = SignatureScheme(target_bytes = 4*64*64//8) 
    signer_private_key, signer_public_key = SignatureScheme.generate_keys(2187)
    
    # 1. Create the codeword tensor.
    codeword = scheme.create(signer_private_key, message)
    print(f"Codeword tensor created with shape: {codeword.shape}")

    # 2. Simulate a noisy channel by corrupting the tensor.
    noisy_codeword = codeword.clone()
    total_bits = noisy_codeword.numel()
    
    # Flip 500 random bits in the tensor ({1, -1} values).
    bits_to_flip = 500
    flip_indices = torch.randperm(total_bits)[:bits_to_flip]
    noisy_codeword.view(-1)[flip_indices] *= -1
    print(f"Simulating noisy channel: Flipped {bits_to_flip} bits.")

    # 3. Decode the *noisy* codeword and verify the signature.
    # This should succeed because RS code can handle the errors.
    is_valid_after_noise = scheme.decode_and_verify(signer_public_key, message, noisy_codeword)
    print(f"Verification result (on noisy data): {is_valid_after_noise}")
    assert is_valid_after_noise

    print("\nAll tests passed.")