import hashlib
import os
import torch
import numpy as np
from py_ecc.bls import G2ProofOfPossession as bls

try:
    from pyldpc import make_ldpc, encode, decode, get_message
except ImportError:
    print("Please install the pyldpc library: pip install pyldpc")
    exit()


class SignatureScheme:
    """
    Implements a scheme using LDPC codes with an interleaver
    to handle both random and burst errors.
    """

    def __init__(self, target_bytes: int = 4*64*64//8):
        """
        Initializes the scheme with an LDPC code and a fixed interleaver.
        Ensures all randomness is removed for reproducibility.
        """
        # --- Set all random seeds for determinism ---
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        # --- LDPC Code Parameters ---
        self.SIG_LEN_BYTES = 96
        # Set LDPC code rate to 1/4: d_v=6, d_c=8 (rate = 1 - d_v/d_c = 0.25)
        self.LDPC_CODEWORD_BITS = 3072  # 96 bytes * 8 = 768 bits for signature, so 768 / 0.25 = 3072 bits

        if target_bytes * 8 < self.LDPC_CODEWORD_BITS:
            raise ValueError("Target bytes is too small to hold the LDPC codeword.")

        self.H, self.G = make_ldpc(self.LDPC_CODEWORD_BITS, d_v=6, d_c=8, systematic=True, sparse=True)
        self.k = self.G.shape[1]
        self.TARGET_BYTES = target_bytes
        self.TOTAL_BITS = target_bytes * 8

        # --- NEW: Create a fixed interleaver and deinterleaver map ---
        rng = np.random.RandomState(42)
        self.interleaver_map = rng.permutation(self.LDPC_CODEWORD_BITS)
        self.deinterleaver_map = np.argsort(self.interleaver_map)
        
        self._original_payload_for_comparison = None

    @staticmethod
    def generate_keys(private_key=2187) -> tuple[int, bytes]:
        public_key = bls.SkToPk(private_key)
        return private_key, public_key

    def create(self, private_key: int, message: bytes) -> torch.Tensor:
        """
        Creates a payload, encodes it with LDPC, and interleaves the result.
        All randomness is removed for reproducibility.
        """
        # --- Set all random seeds for determinism ---
        import random
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        # 1. Sign the message
        signature_c = bls.Sign(private_key, message)
        print(f"Original signature (hex): {signature_c.hex()}")
        # 2. Prepare message bits
        ldpc_message_bits = np.zeros(self.k, dtype=np.uint8)
        signature_bits = np.unpackbits(np.frombuffer(signature_c, dtype=np.uint8))
        ldpc_message_bits[:len(signature_bits)] = signature_bits
        # 3. Encode with LDPC to get the codeword
        ldpc_codeword_floats = encode(self.G, ldpc_message_bits, snr=10000) # High SNR for deterministic output
        # --- MODIFIED: Apply the interleaver ---
        interleaved_codeword = ldpc_codeword_floats[self.interleaver_map]
        # 4. Construct the full payload
        payload_to_mask_tensor = torch.ones(self.TOTAL_BITS, dtype=torch.float64)
        codeword_tensor = torch.from_numpy(interleaved_codeword).double()
        payload_to_mask_tensor[:self.LDPC_CODEWORD_BITS] = codeword_tensor
        # 5. Mask with one-time pad
        one_time_pad_bytes = self._hash_long(message, self.TARGET_BYTES)
        one_time_pad_bits = np.unpackbits(np.frombuffer(one_time_pad_bytes, dtype=np.uint8))
        one_time_pad_tensor = torch.from_numpy(1.0 - 2.0 * one_time_pad_bits).double()
        masked_tensor = payload_to_mask_tensor * one_time_pad_tensor
        self._original_payload_for_comparison = payload_to_mask_tensor.clone()
        return masked_tensor

    def decode_and_verify(self, public_key: bytes, message: bytes, noisy_tensor: torch.Tensor) -> bool:
        """
        De-interleaves a noisy tensor, decodes with LDPC, and verifies.
        """
        # 1. Unmask the payload
        one_time_pad_bytes = self._hash_long(message, self.TARGET_BYTES)
        one_time_pad_bits = np.unpackbits(np.frombuffer(one_time_pad_bytes, dtype=np.uint8))
        one_time_pad_tensor = torch.from_numpy(1.0 - 2.0 * one_time_pad_bits).double()
        noisy_unmasked_tensor = noisy_tensor.view(-1).double() * one_time_pad_tensor

        # 2. Extract the noisy LDPC codeword
        noisy_interleaved_codeword = noisy_unmasked_tensor[:self.LDPC_CODEWORD_BITS].numpy()

        # --- MODIFIED: Apply the deinterleaver first ---
        # This spreads the burst errors out, making them appear random to the decoder.
        noisy_deinterleaved_codeword = noisy_interleaved_codeword[self.deinterleaver_map]
        
        # 3. Decode using LDPC. The SNR must match the actual channel noise.
        # Assuming SNR=1.0 for this high-noise example.
        decoded_codeword = decode(self.H, noisy_deinterleaved_codeword, snr=1, maxiter=600)
        
        if decoded_codeword is None:
            print("LDPC decode failed - returned None")
            return False
        
        # 4. Extract the message from the decoded codeword
        decoded_message_bits = get_message(self.G, decoded_codeword)
        
        # 5. Reconstruct and verify the signature
        signature_bits = decoded_message_bits[:self.SIG_LEN_BYTES * 8]
        recovered_signature_c = np.packbits(signature_bits).tobytes()
        print(f"Recovered signature (hex): {recovered_signature_c.hex()}")
        verification_result = bls.Verify(public_key, message, recovered_signature_c)
        
        return verification_result

    @staticmethod
    def _hash_long(message: bytes, length: int) -> bytes:
        hasher = hashlib.shake_256()
        hasher.update(message)
        return hasher.digest(length)

# --- Example Usage ---
if __name__ == "__main__":
    # --- Setup ---
    scheme = SignatureScheme()
    message = b"hi"
    private_key, public_key = SignatureScheme.generate_keys(729)

    # --- NEW: Test Case for Burst Noise ---
    print("--- Testing with controlled BURST noise ---")
    
    # 1. Create the ideal, masked tensor
    golden_masked_tensor = scheme.create(private_key, message)
    torch.save(golden_masked_tensor, "codeword.pt")
    # 2. Simulate a burst of errors
    # Create a noise vector that is mostly zeros, with one concentrated burst
    burst_length = 200  # A long burst of 800 consecutive errors
    burst_start = 1000
    burst_intensity = 2.0 # Strong enough to guarantee bit flips
    
    noise = torch.zeros_like(golden_masked_tensor, dtype=torch.float64)
    # The burst is a block of high-intensity random noise
    noise[burst_start : burst_start + burst_length] = torch.randn(burst_length, dtype=torch.float64) * burst_intensity

    # 3. Add the burst noise to the tensor
    noisy_tensor = golden_masked_tensor + noise
    print(f"Added a burst of noise (length={burst_length}) to the golden tensor.")

    # 4. Attempt to verify the payload corrupted by the burst
    is_valid_burst_noise = scheme.decode_and_verify(public_key, message, noisy_tensor)
    print(f"Verification successful with burst noise: {is_valid_burst_noise}")
    assert is_valid_burst_noise, "Verification with BURST noise FAILED!"
    
    print("\nControlled burst noise test passed.")
    print("--- TEST CASE 3: Decoding from 'bryces_latent.pt' ---")
    latent = torch.load('decode_testing/0_latent.pt', map_location='cpu')
    print(f"Loaded latent tensor of shape: {latent.shape}")
    is_valid_from_latent = scheme.decode_and_verify(public_key=public_key, message=message, noisy_tensor=latent)
    print(f"Verification successful from latent: {is_valid_from_latent}\n")