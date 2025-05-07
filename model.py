import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
  def __init__(self,
               in_channel: int,
               out_channel: int = None,
               group_channel: int = 32,
               drop: float = 0,
               eps: float = 1e-5,
              #  down: bool = False,
              #  up: bool = False
               ):
    super().__init__()
    
    out_channel = in_channel if out_channel is None else out_channel 
    self.out_channel = out_channel
    self.norm1 = nn.GroupNorm(num_groups= group_channel, num_channels=in_channel, eps = eps)
    self.norm2 = nn.GroupNorm(num_groups= group_channel, num_channels=out_channel, eps = eps)
    
    self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
    
    self.act = nn.SiLU()
    self.dropout = nn.Dropout(p=drop)
    
    self.residual_layer = nn.Conv2d(in_channel, out_channel, kernel_size=1) if in_channel != out_channel else nn.Identity()
    
  def forward(self, x: torch.Tensor):
    # x -> (b,c,h,w)
    h = self.conv1(self.act(self.norm1(x))) # (b,c',h,w)
    h = self.dropout(h)
    h = self.conv2(self.act(self.norm2(h)))
    
    x = self.residual_layer(x) + h
    return x
  
class Downsample(nn.Module):
  def __init__(self,
               in_channel: int,
               out_channel: int = None,
               use_conv:bool = False,
               kernel:int = 3,
               stride:int = 2
               ):
    super().__init__()
    
    out_channel = in_channel if out_channel is None else out_channel 
    self.out_channel = out_channel
    self.padding = kernel//2 #smooth padding
    self.use_conv = use_conv
    
    if use_conv:
      self.down_layer = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=self.padding)
    else:
      self.down_layer = nn.AvgPool2d(kernel_size=stride, stride=stride)
  
  def forward(self, x: torch.Tensor):
    #x -> (b,c,h,w)
    if not self.use_conv:
      pad = (0, 1, 0, 1)
      x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
    return self.down_layer(x)

class Upsample(nn.Module):
  def __init__(self,
               in_channel: int,
               out_channel: int = None,
               use_conv:bool = False,
               use_conv_tranpose:bool = False,
               interpolate:bool = True
               ):
    super().__init__()
    
    out_channel = in_channel if out_channel is None else out_channel 
    self.out_channel = out_channel
    self.use_conv = use_conv
    self.use_conv_tranpose = use_conv_tranpose
    self.interpolate = interpolate
    
    self.layer = None
    if use_conv:
      self.layer = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
    elif use_conv_tranpose:
      self.layer = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1)
  
  def forward(self, x: torch.Tensor):
    #x -> (b,c,h,w)
    
    #simply using the tranpose conv
    if self.use_conv_tranpose:
      return self.layer(x)
    
    if self.interpolate: #use interpolate and then option to use conv
      x = nn.functional.interpolate(x, scale_factor=2, mode = 'nearest')
      
    if self.use_conv:
      x = self.layer(x)

    return x

class EncoderBlock(nn.Module):
  def __init__(self,
               num_res_layers:int,
               in_channel:int,
               out_channel:int,
               group_channel:int,
               drop:int = 0,
               eps:float = 1e-5,
               down_layer:bool = True,
               ):
    super().__init__()
    
    self.down_layer = down_layer
    self.layers = nn.ModuleList([])
    
    for idx in range(num_res_layers):
      in_channel = in_channel if idx == 0 else out_channel
      self.layers.append(ResnetBlock(in_channel,out_channel,group_channel,drop, eps))

    if self.down_layer:
      self.down_block = Downsample(out_channel, use_conv=True)
    
  def forward(self, x:torch.Tensor):
    
    for layer in self.layers:
      x = layer(x)
    
    if self.down_layer:
      x = self.down_block(x)
    return x
    
class DecoderBlock(nn.Module):
  def __init__(self,
               num_res_layers:int,
               in_channel:int,
               out_channel:int,
               group_channel:int,
               drop:int = 0,
               eps:float = 1e-5,
               up_layer:bool = True
               ):
    super().__init__()
    
    self.up_layer = up_layer
    
    self.layers = nn.ModuleList([])
    
    for idx in range(num_res_layers):
      in_channel = in_channel if idx == 0 else out_channel
      self.layers.append(ResnetBlock(in_channel,out_channel,group_channel,drop, eps))

    if self.up_layer:
      self.up_block = Upsample(out_channel, use_conv=True, interpolate=True)
    
  def forward(self, x:torch.Tensor):
    
    for layer in self.layers:
      x = layer(x)
    
    if self.up_layer:
      x = self.up_block(x)
    return x

#more type of block can be added here like attention_block
class BottleNeck(nn.Module):
  def __init__(self,
              num_res_layers:int,
              in_channel:int,
              out_channel:int,
              group_channel:int,
              drop:int = 0,
              eps:float = 1e-5):
    super().__init__()
    
    self.layers = nn.ModuleList([])
    
    for idx in range(num_res_layers):
      in_channel = in_channel if idx == 0 else out_channel
      self.layers.append(ResnetBlock(in_channel,out_channel,group_channel,drop, eps))
    
  def forward(self, x:torch.Tensor):
    
    for layer in self.layers:
      x = layer(x)
    return x
    

#Encoder block + bottleneck block  
class Encoder(nn.Module):
    def __init__(self,
                  block_out_channels:list[int],
                  input_channel:int = 3,
                  output_channel:int = 4, #latent channels
                  num_res_layers:int = 2,
                  group_channel: int =32,
                  drop: float = 0,
                  eps:float = 1e-5
                  ):
      super().__init__()
      
      out_channel = block_out_channels[0]
      in_channel = input_channel
      #first layer for projection
      self.first_layer = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
      
      #encoder part
      self.encoder_blocks = nn.ModuleList([])
      for idx,_ in enumerate(block_out_channels):
        in_channel = out_channel
        # out_channel = block_out_channels[idx + 1] if idx < len(block_out_channels)-1 else in_channel
        out_channel = block_out_channels[idx] 
        last_block = idx == len(block_out_channels)-1
        self.encoder_blocks.append(EncoderBlock(num_res_layers,in_channel,out_channel,group_channel,drop,eps,not last_block))

      #bottleneck part
      final_channel = block_out_channels[-1]
      self.bottleneck = BottleNeck(num_res_layers=num_res_layers,in_channel=final_channel,out_channel=final_channel,group_channel= group_channel,drop=drop,eps=eps)
      
      self.latent_out = nn.Sequential(
        nn.GroupNorm(num_groups=32, num_channels=block_out_channels[-1]),
        nn.SiLU(),
        nn.Conv2d(block_out_channels[-1], 2 * output_channel, kernel_size=3, padding=1)
      )
      
    def forward(self, x):
      
      x = self.first_layer(x)
      for encoder_block in self.encoder_blocks:
        x = encoder_block(x)
      
      x = self.bottleneck(x)
      return self.latent_out(x)
      

class Decoder(nn.Module):
    def __init__(self,
                  block_out_channels:list[int],
                  input_channel:int = 4, #latent channels
                  output_channel:int = 3, 
                  num_res_layers:int = 2,
                  group_channel: int = 32,
                  drop:float = 0,
                  eps:float = 1e-5
                  ):
      super().__init__()
      
      block_out_channels = block_out_channels[::-1]
      
      out_channel = block_out_channels[0]
      in_channel = input_channel 
      #first layer for projection
      self.first_layer = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
      
      #bottleneck part
      final_channel = block_out_channels[0]
      self.bottleneck = BottleNeck(num_res_layers=num_res_layers,in_channel=final_channel,out_channel=final_channel,group_channel= group_channel,drop=drop,eps=eps)
      
      #encoder part
      self.decoder_blocks = nn.ModuleList([])
      for idx,_ in enumerate(block_out_channels):
        in_channel = out_channel
        # out_channel = block_out_channels[idx + 1] if idx < len(block_out_channels)-1 else in_channel
        out_channel = block_out_channels[idx]
        last_block = idx == len(block_out_channels)-1
        self.decoder_blocks.append(DecoderBlock(num_res_layers,in_channel,out_channel,group_channel,drop,eps,not last_block))

      self.image_out = nn.Sequential(
        nn.GroupNorm(num_groups=32, num_channels=block_out_channels[-1]),
        nn.SiLU(),
        nn.Conv2d(block_out_channels[-1], output_channel, kernel_size=3, padding=1),
        nn.Tanh()
      )
      
    def forward(self, x):
      
      x = self.first_layer(x)
      x = self.bottleneck(x)
      for decoder_block in self.decoder_blocks:
        x = decoder_block(x)
        
      return self.image_out(x)


class VAE(nn.Module):
  def __init__(self, config):
    super().__init__()
    block_out_channel = config.block_out_channel
    input_channel = config.input_channel
    output_channel = config.output_channel
    latent_channel = config.latent_channel
    num_res_layers = config.num_res_layers
    group_channels = config.group_channels
    
    self.encoder = Encoder(block_out_channels= block_out_channel,
                           input_channel = input_channel,
                           output_channel= latent_channel,
                           num_res_layers= num_res_layers,
                           group_channel= group_channels
                           )
    self.decoder = Decoder(
                          block_out_channels= block_out_channel,
                          input_channel = latent_channel,
                          output_channel= output_channel,
                          num_res_layers= num_res_layers,
                          group_channel= group_channels
                          )
    
  def encode(self, x: torch.Tensor):
    h = self.encoder(x)
    #considering the covariance as diagonal (independent variables)
    mu, log_var = torch.split(h, h.shape[1]//2, dim = 1) #splitting over the channel_dim
    var = torch.exp(log_var)
    std = torch.exp(0.5 * log_var)
    z = mu + std * torch.randn_like(mu)
    return z, mu, var
  
  def decode(self, z: torch.Tensor):
    x = self.decoder(z)
    return x
    
  def forward(self, x: torch.Tensor):
    
    z, mu, var= self.encode(x)
    x = self.decode(z)
    return x, mu, var


from config import VAEConfig
config = VAEConfig()
if __name__ == "__main__":
  x = torch.randn((2,3,256,256))
  # vae = VAE([128,256,512,512],input_channel=3,output_channel=3,latent_channel=4,num_res_layers=2,group_channels=32)
  vae = VAE(config)
  x, mu, var = vae(x)
  parameters = sum([p.numel() for p in vae.parameters()])
  print(f"parameters: {parameters}")
  print(f"constructed_shape: {x.shape}")
  print(f"mean_shape: {mu.shape}")
  print(f"var_shape: {var.shape}")
  print(x.min(), x.max())




      