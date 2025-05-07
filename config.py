from dataclasses import dataclass

@dataclass
class VAEConfig:
    block_out_channel: tuple[int] = (64,64,128,256)
    input_channel:int = 3
    output_channel:int = 3
    latent_channel:int = 8
    num_res_layers:int = 2
    group_channels:int = 16
    lr:float = 1e-4
    beta:float = 0.02
    epochs:int = 500
    batch: int = 32
    image_size:int = 128
    folder_path:str = "F:\dataset"
    train_split:float = 0.9
    
    
  