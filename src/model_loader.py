from src.clip import CLIP
from src.encoder import VAEEncoder
from src.decoder import VAEDecoder
from src.diffusion import Diffusion

import src.model_converter as model_converter


def preload_models_from_standard_weights(checkpoint_path, device):
    state_dict = model_converter.load_from_standard_weights(checkpoint_path, device)
    print(f'Loaded weights from {checkpoint_path}')
    print(state_dict.keys())

    encoder = VAEEncoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAEDecoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        "clip": clip,
        "encoder": encoder,
        "decoder": decoder,
        "diffusion": diffusion
    }
