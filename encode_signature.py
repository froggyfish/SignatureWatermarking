import hashlib
import os
import torch
import numpy as np
from py_ecc.bls import G2ProofOfPossession as bls
import src.pseudogaussians as prc_gaussians


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
        # Using a rate 1/8 code (d_v=7, d_c=8 gives rate ~0.125)
        # Need codeword length divisible by 8 and large enough for 768-bit signature
        # With rate 1/8, we need at least 768 * 8 = 6144 bits for the codeword
        self.LDPC_CODEWORD_BITS = 8192  # Increased to accommodate 768-bit signature

        if target_bytes * 8 < self.LDPC_CODEWORD_BITS:
            raise ValueError("Target bytes is too small to hold the LDPC codeword.")

        # Create the LDPC generator (G) and parity-check (H) matrices.
        # Use a rate 1/8 code for better error correction (d_v=7, d_c=8 gives rate ~0.125)
        # Note: d_c must divide the codeword length, so we'll use d_c=8 (8192/8=1024)
        self.H, self.G = make_ldpc(self.LDPC_CODEWORD_BITS, d_v=7, d_c=8, systematic=True, sparse=True)
        
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
        
        # Debug: Print original signature information
        print("ORIGINAL SIGNATURE DEBUG:")
        print(f"Original signature length: {len(signature_c)} bytes")
        print(f"Original signature first 10 bytes: {signature_c[:10]}")
        print(f"Original signature bits: {np.unpackbits(np.frombuffer(signature_c, dtype=np.uint8))[:20]}")

        # 2. Prepare the message block for LDPC encoding at the bit level.
        ldpc_message_bits = np.zeros(self.k, dtype=np.uint8)
        signature_bits = np.unpackbits(np.frombuffer(signature_c, dtype=np.uint8))
        ldpc_message_bits[:len(signature_bits)] = signature_bits

        # 3. Encode the message bits with LDPC. The output is noisy floats.
        ldpc_codeword_floats = encode(self.G, ldpc_message_bits, snr= 1000)
        
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
        
        # Print the original payload for comparison
        print("ORIGINAL PAYLOAD (before masking):")
        print("Payload shape:", payload_to_mask_tensor.shape)
        print("Payload first 20 values:", payload_to_mask_tensor[:20])
        print("Payload last 20 values:", payload_to_mask_tensor[-20:])
        print("Unique values in payload:", torch.unique(payload_to_mask_tensor))
        
        # Store the original payload for comparison in decode_and_verify
        self._original_payload_for_comparison = payload_to_mask_tensor.clone()
        
        return masked_tensor

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
        
        # Use soft values for LDPC decoding (better than hard quantization)
        # noisy_unmasked_tensor = torch.sign(noisy_unmasked_tensor)  # Comment out hard quantization
        
        # Alternative: Use soft values for LDPC decoding (might work better)
        # noisy_unmasked_tensor_soft = noisy_tensor.view(-1).double() * one_time_pad_tensor
        print("Using soft values for LDPC decoding")
        
        # Print the unmasked payload for comparison
        print("UNMASKED PAYLOAD (after quantization):")
        print("Unmasked payload shape:", noisy_unmasked_tensor.shape)
        print("Unmasked payload first 20 values:", noisy_unmasked_tensor[:20])
        print("Unmasked payload last 20 values:", noisy_unmasked_tensor[-20:])
        print("Unique values in unmasked payload:", torch.unique(noisy_unmasked_tensor))
        
        # Calculate similarity metrics
        if hasattr(self, '_original_payload_for_comparison'):
            original_payload = self._original_payload_for_comparison
            # Ensure same length for comparison
            min_len = min(len(original_payload), len(noisy_unmasked_tensor))
            original_subset = original_payload[:min_len]
            unmasked_subset = noisy_unmasked_tensor[:min_len]
            
            # Calculate accuracy (percentage of matching signs)
            matching_signs = torch.sum(original_subset * unmasked_subset > 0).item()
            accuracy = matching_signs / min_len
            
            # Calculate average distance
            distance = torch.mean(torch.abs(original_subset - unmasked_subset)).item()
            
            print(f"PAYLOAD COMPARISON:")
            print(f"Accuracy (matching signs): {accuracy:.4f} ({matching_signs}/{min_len})")
            print(f"Average distance: {distance:.4f}")
            print(f"Original payload unique values: {torch.unique(original_subset)}")
            print(f"Unmasked payload unique values: {torch.unique(unmasked_subset)}")
        
        # 3. Extract the noisy LDPC codeword (soft float values).
        noisy_codeword_floats = noisy_unmasked_tensor[:self.LDPC_CODEWORD_BITS].numpy()
        
        # Debug: Print LDPC codeword information
        print("LDPC CODEWORD DEBUG:")
        print(f"LDPC codeword length: {len(noisy_codeword_floats)}")
        print(f"LDPC codeword first 20 values: {noisy_codeword_floats[:20]}")
        print(f"LDPC codeword last 20 values: {noisy_codeword_floats[-20:]}")
        print(f"LDPC codeword unique values: {np.unique(noisy_codeword_floats)}")
        print(f"LDPC codeword mean: {np.mean(noisy_codeword_floats):.4f}")
        print(f"LDPC codeword std: {np.std(noisy_codeword_floats):.4f}")
        
        # 4. Decode using LDPC with the soft float values.
        # Try higher SNR and more iterations for better error correction
        decoded_codeword = decode(self.H, noisy_codeword_floats, snr=1, maxiter=600)
        
        # Debug: Print decoded codeword information
        print("DECODED CODEWORD DEBUG:")
        print(f"Decoded codeword type: {type(decoded_codeword)}")
        if decoded_codeword is not None:
            print(f"Decoded codeword shape: {decoded_codeword.shape}")
            print(f"Decoded codeword first 20 values: {decoded_codeword[:20]}")
            print(f"Decoded codeword unique values: {np.unique(decoded_codeword)}")
        else:
            print("LDPC decode failed - returned None")
            return False
        
        # 5. Extract the original message from the systematic codeword.
        decoded_message_bits = get_message(self.G, decoded_codeword)
        
        # Debug: Print decoded message information
        print("DECODED MESSAGE DEBUG:")
        print(f"Decoded message bits length: {len(decoded_message_bits)}")
        print(f"Decoded message bits first 20: {decoded_message_bits[:20]}")
        print(f"Decoded message bits unique values: {np.unique(decoded_message_bits)}")
        
        # 6. Extract the original signature bits from the decoded message block.
        signature_bits = decoded_message_bits[:self.SIG_LEN_BYTES * 8]
        recovered_signature_c = np.packbits(signature_bits).tobytes()
        
        # Debug: Print signature extraction information
        print("SIGNATURE EXTRACTION DEBUG:")
        print(f"Signature bits length: {len(signature_bits)}")
        print(f"Signature bits first 20: {signature_bits[:20]}")
        print(f"Signature bits last 20: {signature_bits[-20:]}")
        print(f"Recovered signature length: {len(recovered_signature_c)} bytes")
        print(f"Recovered signature first 10 bytes: {recovered_signature_c[:10]}")
        
        # 7. Verify the recovered, pristine signature.
        verification_result = bls.Verify(public_key, message, recovered_signature_c)
        print(f"BLS verification result: {verification_result}")
        
        return verification_result


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
    print('started')
    # --- Setup ---
    scheme = SignatureScheme()
    message = b"hi"
    private_key, public_key = SignatureScheme.generate_keys(729)

    # --- TEST CASE 1: Verification from your loaded latent tensor ---
    # print("--- TEST CASE 1: Verifying from loaded 'latent.pt' ---")
    
    # latent = torch.load('decode_testing/bryces_latent.pt', map_location='cpu')
    # print("BLAHBLAHBLAH DECODED LATENT", latent.view(-1))

    # scheme = SignatureScheme()
    # # 2. Generate a key pair for the signer.
    # signer_private_key, signer_public_key = SignatureScheme.generate_keys(729)

    # message = b"hi"
    # codeword = scheme.create(signer_private_key, message)
    # init_latents = prc_gaussians.sample(codeword)
    # print("ORIGINAL LATENT", init_latents)
    # flattened_latent = latent.view(-1)
    # counter = 0
    # distance = 0
    # for i in range(len(init_latents)):
    #     if(init_latents[i]*flattened_latent[i] >= 0):
    #         counter += 1
    #     distance += abs(init_latents[i] - flattened_latent[i])
    # print("accuracy", counter/len(init_latents))
    # print("avg distance", distance/len(init_latents))

    # # The decode function now takes the tensor directly
    # print(f"Attempting to decode and verify latent tensor of shape: {latent.shape}")
    # is_valid_from_latent = scheme.decode_and_verify(public_key=public_key, message=message, noisy_tensor=latent)
    # print(f"Verification successful from latent: {is_valid_from_latent}\n")


    # --- TEST CASE 2: Controlled High-Noise Test with LDPC ---
    print("--- TEST CASE 2: Verifying with controlled, HIGH noise ---")
    
    golden_codeword_tensor = scheme.create(private_key, message)
    
    # Reshape the flat tensor to (4, 64, 64) for the noise test
    golden_codeword_reshaped = golden_codeword_tensor.view(4, 64, 64)
    
    # Corrupt the tensor by adding Gaussian noise. This is a more realistic
    # simulation of a noisy channel than flipping bits.
    # FIX: Ensure noise is also float64.
    noise = torch.randn_like(golden_codeword_reshaped, dtype=torch.float64) * 1 # Add noise with std dev 0.2
    # for item in golden_codeword_reshaped:
    #     for i in item:
    #         print(i)
    noisy_tensor = golden_codeword_reshaped + noise
    # print(golden_codeword_tensor.shape)

    print(f"Added Gaussian noise to the golden tensor.")

    # Attempt to verify the heavily corrupted payload.
    is_valid_controlled_noise = scheme.decode_and_verify(public_key=public_key, message=message, noisy_tensor=noisy_tensor)
    print(f"Verification successful with high noise: {is_valid_controlled_noise}")
    assert is_valid_controlled_noise, "Verification with high noise FAILED!"
    
    print("\nControlled high-noise test passed.")
