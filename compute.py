import torch, os, numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, padding_mode):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode),
        )

    def forward(self, x):
        #Elementwise Sum (ES)
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, num_features=32, num_residuals=4):
        super().__init__()
        self.padding_mode = "zeros"

        self.initial_down = nn.Sequential(
            #k7n32s1
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        #Down-convolution
        self.down1 = nn.Sequential(
            #k3n32s2   256, 256, 32
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=2, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n64s1   128, 128, 64
            nn.Conv2d(num_features, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.down2 = nn.Sequential(
            #k3n64s2
            nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=2, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #k3n128s1  64, 64, 128
            nn.Conv2d(num_features*2, num_features*4, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        #Bottleneck: 4 residual blocks => 4 times [K3n128s1]  64, 64, 128
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode) for _ in range(num_residuals)]
        )

        #Up-convolution
        self.up1 = nn.Sequential(
            #k3n128s1   64, 64, 128
            nn.Conv2d(num_features*4, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.up2 = nn.Sequential(
            #k3n64s1
            nn.Conv2d(num_features*2, num_features*2, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #k3n64s1 (should be k3n32s1?)  128, 128, 64
            nn.Conv2d(num_features*2, num_features, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.last = nn.Sequential(
            #k3n32s1
            nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1, padding_mode=self.padding_mode),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            #k7n3s1   256, 256, 32
            nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode=self.padding_mode)
        )

    def forward(self, x):
        x1 = self.initial_down(x)
        x2 = self.down1(x1)
        x = self.down2(x2)
        x = self.res_blocks(x)
        x = self.up1(x)
        #Resize Bilinear
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners = False)
        x = self.up2(x + x2) 
        #Resize Bilinear
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners = False)
        x = self.last(x + x1)
        #TanH
        return torch.tanh(x)

def resize_crop(image):
  target_height = 256
  target_width = 256
  w, h = image.size

  if h and w == 256:
    print("No resizing needed")
    return image
  elif (h!=256) or (w!= 256):
    diff_height = abs(target_height - h)
    if diff_height > 256:
      h = h - diff_height
    elif diff_height < 256:
      h = diff_height + h
    else:
      h = diff_height
    
    diff_width = abs(target_width - w)
    if diff_width > 256:
      w = w - diff_width
    elif diff_width < 256:
      w = diff_width + w
    else:
      w = diff_width
  image = image.resize((w, h), Image.ANTIALIAS)
  print("Done resizing")
  return image

DEVICE = 'cpu'
LEARNING_RATE = 2e-4

trained_network_name = "201_last_gen.pth.tar"

def load_checkpoint(model, optimizer, lr, path):
    #print("=> Loading checkpoint")
    if (os.path.isfile(path)):
        checkpoint = torch.load(path, map_location = DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # If we don't do this then it will just have learning rate of old checkpoint
        # and it will lead to many hours of debugging \:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        #print("Checkpoint file " + str(path) + " loaded.")
        loaded = True
    else:
        print("Checkpoint file " + str(path) + " not found. Not loading checkpoint.")
        loaded = False
    return model, optimizer, loaded

gen = Generator(img_channels=3).to(DEVICE)
opt_gen = torch.optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
load_checkpoint(gen, opt_gen, LEARNING_RATE, path = trained_network_name)

def generate(image):
    test_image = Image.open(image).convert("RGB")
    #print(type(test_image))
    test_image = resize_crop(test_image)
    test_image = transforms.ToTensor()(test_image).to(DEVICE).unsqueeze_(0)
    #print(type(test_image))
    y_fake = gen(test_image)
    fake_image_pil = y_fake[0].cpu().squeeze_(0).permute(1, 2, 0).detach().numpy()
    fake_image = np.clip(fake_image_pil, 0, 1)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(BASE_DIR, 'static/temp_images')
    save_path = os.path.join(temp_dir , 'fake.jpg')
    processed_image = Image.fromarray((fake_image * 255).astype(np.uint8))
    processed_image.save(save_path)
    return Image.fromarray((fake_image * 255).astype(np.uint8))



if __name__ == '__main__':
   generate('00010.jpg')
