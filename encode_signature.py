import hashlib
import os
import torch
import numpy as np
from py_ecc.bls import G2ProofOfPossession as bls

# --- Installation ---
# You will need to install py_ecc, pyldpc, and numpy:
# pip install py_ecc pyldpc numpy
try:
    from pyldpc import make_ldpc, encode, decode, get_message
except ImportError:
    print("Please install the pyldpc library: pip install pyldpc")
    exit()


class SignatureScheme:
    """
    Implements a scheme to create and verify a payload that combines a BLS
    signature with a long-form hash of the original message.
    Includes Low-Density Parity-Check (LDPC) codes to handle high-noise channels.
    """

    def __init__(self, target_bytes: int = 4*64*64//8):
        """
        Initializes the scheme with a target payload size and LDPC code.

        Args:
            target_bytes: The total desired length of the final payload in bytes.
        """
        # --- LDPC Code Parameters ---
        self.SIG_LEN_BYTES = 96  # BLS signature length is fixed
        
        # Define the total number of bits in the LDPC codeword.
        # This will be protected by the error correction.
        self.LDPC_CODEWORD_BITS = 2048

        if target_bytes * 8 < self.LDPC_CODEWORD_BITS:
            raise ValueError("Target bytes is too small to hold the LDPC codeword.")

        # Create the LDPC generator (G) and parity-check (H) matrices.
        self.H, self.G = make_ldpc(self.LDPC_CODEWORD_BITS, d_v=2, d_c=4, systematic=True, sparse=True)
        
        # The number of message bits k is the number of COLUMNS in the G matrix.
        self.k = self.G.shape[1]
        self.TARGET_BYTES = target_bytes
        self.TOTAL_BITS = target_bytes * 8

    @staticmethod
    def generate_keys(private_key=2187) -> tuple[int, bytes]:
        """
        Generates a valid BLS public key using the given private key.
        """
        public_key = bls.SkToPk(private_key)
        return private_key, public_key

    def create(self, private_key: int, message: bytes) -> torch.Tensor:
        """
        Creates a verifiable payload, encodes it with LDPC,
        and returns a bipolar tensor.
        """
        # 1. Sign the message to get the 96-byte signature 'c'.
        signature_c = bls.Sign(private_key, message)

        # 2. Prepare the message block for LDPC encoding at the bit level.
        ldpc_message_bits = np.zeros(self.k, dtype=np.uint8)
        signature_bits = np.unpackbits(np.frombuffer(signature_c, dtype=np.uint8))
        ldpc_message_bits[:len(signature_bits)] = signature_bits

        # 3. Encode the message bits with LDPC. The output is noisy floats.
        ldpc_codeword_floats = encode(self.G, ldpc_message_bits, snr=100)
        
        # 4. Construct the full payload to be masked, in the bipolar {-1, 1} domain.
        #    FIX: Ensure consistent dtype of float64 for compatibility with Numba.
        payload_to_mask_tensor = torch.ones(self.TOTAL_BITS, dtype=torch.float64)
        # Convert codeword floats to a tensor
        codeword_tensor = torch.from_numpy(ldpc_codeword_floats) # This will be float64
        # Place the codeword at the beginning of the payload tensor
        payload_to_mask_tensor[:self.LDPC_CODEWORD_BITS] = codeword_tensor

        # 5. Generate the one-time pad and convert it to the bipolar domain.
        one_time_pad_bytes = self._hash_long(message, self.TARGET_BYTES)
        one_time_pad_bits = np.unpackbits(np.frombuffer(one_time_pad_bytes, dtype=np.uint8))
        # FIX: Ensure consistent dtype of float64. .double() is equivalent to .to(torch.float64)
        one_time_pad_tensor = torch.from_numpy(1.0 - 2.0 * one_time_pad_bits).double()

        # 6. Mask the payload using multiplication in the bipolar domain (equivalent to XOR).
        masked_tensor = payload_to_mask_tensor * one_time_pad_tensor
        
        return masked_tensor.view(4, 64, 64)

    def decode_and_verify(self, public_key: bytes, message: bytes, noisy_tensor: torch.Tensor) -> bool:
        """
        Decodes a noisy tensor, corrects errors using LDPC,
        and verifies the underlying signature.
        """
        # 1. Regenerate the one-time pad and convert it to the bipolar domain.
        one_time_pad_bytes = self._hash_long(message, self.TARGET_BYTES)
        one_time_pad_bits = np.unpackbits(np.frombuffer(one_time_pad_bytes, dtype=np.uint8))
        # FIX: Ensure consistent dtype of float64.
        one_time_pad_tensor = torch.from_numpy(1.0 - 2.0 * one_time_pad_bits).double()

        # 2. Unmask the payload by multiplying with the one-time pad tensor.
        #    Ensure the noisy tensor is also float64 before the operation.
        noisy_unmasked_tensor = noisy_tensor.view(-1).double() * one_time_pad_tensor

        # 3. Extract the noisy LDPC codeword (soft float values).
        noisy_codeword_floats = noisy_unmasked_tensor[:self.LDPC_CODEWORD_BITS].numpy()

        # 4. Decode using LDPC with the soft float values.
        decoded_codeword = decode(self.H, noisy_codeword_floats, snr=10)
        
        # 5. Extract the original message from the systematic codeword.
        decoded_message_bits = get_message(self.G, decoded_codeword)
        
        # 6. Extract the original signature bits from the decoded message block.
        signature_bits = decoded_message_bits[:self.SIG_LEN_BYTES * 8]
        recovered_signature_c = np.packbits(signature_bits).tobytes()

        # 7. Verify the recovered, pristine signature.
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

    # --- Setup ---
    scheme = SignatureScheme()
    message = b"hi"
    private_key, public_key = SignatureScheme.generate_keys(729)

    # --- TEST CASE 1: Verification from your loaded latent tensor ---
    print("--- TEST CASE 1: Verifying from loaded 'latent.pt' ---")
    
    latent = torch.load('decode_testing/bryces_latent.pt', map_location='cpu')
    
    # The decode function now takes the tensor directly
    print(f"Attempting to decode and verify latent tensor of shape: {latent.shape}")
    is_valid_from_latent = scheme.decode_and_verify(public_key=public_key, message=message, noisy_tensor=latent)
    print(f"Verification successful from latent: {is_valid_from_latent}\n")


    # --- TEST CASE 2: Controlled High-Noise Test with LDPC ---
    print("--- TEST CASE 2: Verifying with controlled, HIGH noise ---")
    
    golden_codeword_tensor = scheme.create(private_key, message)
    
    
    # Corrupt the tensor by adding Gaussian noise. This is a more realistic
    # simulation of a noisy channel than flipping bits.
    # FIX: Ensure noise is also float64.
    noise = torch.randn_like(golden_codeword_tensor, dtype=torch.float64) * 0.3 # Add noise with std dev 0.2
    noisy_tensor = golden_codeword_tensor + noise

    print(f"Added Gaussian noise to the golden tensor.")

    # Attempt to verify the heavily corrupted payload.
    is_valid_controlled_noise = scheme.decode_and_verify(public_key=public_key, message=message, noisy_tensor=noisy_tensor)
    print(f"Verification successful with high noise: {is_valid_controlled_noise}")
    assert is_valid_controlled_noise, "Verification with high noise FAILED!"
    
    print("\nControlled high-noise test passed.")
