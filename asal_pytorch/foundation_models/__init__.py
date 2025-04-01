from .clip import CLIP
# from .blip2 import BLIP2
# from .clip4clip import CLIP4CLIP
from .gemma3 import Gemma3Chat
# from .video_clip import VideoCLIP

def create_foundation_model(fm_name, device):
    """
    Create the foundation model given a foundation model name.
    It has the following methods attached to it:
        - fm.embed_img(img)
        - fm.embed_txt(prompts)
    Some foundation models may not have the embed_text method.

    Possible foundation model names:
        - 'clip': CLIP
    """
    if fm_name=='clip':
        fm = CLIP(device)
    # elif fm_name=='blip2':
    #    fm= BLIP2(device)
    # elif fm_name=='clip4clip':
    #    fm= CLIP4CLIP(device)
    elif fm_name=='gemma3':
        fm= Gemma3Chat(device)
    # elif fm_name=='video_clip':
    #     fm= VideoCLIP(device)
    else:
        raise ValueError(f"Unknown foundation model name: {fm_name}")
    return fm
