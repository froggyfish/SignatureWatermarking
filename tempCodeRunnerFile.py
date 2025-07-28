 from loaded 'latent.pt' ---")
    
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
    # print(f"Verification successful