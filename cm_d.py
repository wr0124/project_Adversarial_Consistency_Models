import os
import numpy as np
from typing import Tuple, List, Union
from torch import Tensor

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.utils import make_grid

from visdom import Visdom
from tqdm import tqdm

from dataclasses import dataclass
from options import parse_opts
from UNet import UNet, UNetConfig
from UNet_discriminator import UNet_discriminator, UNetConfig_discriminator
from consistency_generator import ImprovedConsistencyTraining, pseudo_huber_loss
from inference import ConsistencySamplingAndEditing
from collections import deque
from torcheval.metrics import FrechetInceptionDistance

from datetime import datetime
torch.autograd.set_detect_anomaly(True)

args = parse_opts()
# Decide which device to run on
device = torch.device(args.device_cuda if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
visdom = Visdom(env=args.env)
start_time = datetime.now()
formatted_datetime = start_time.strftime("%Y-%m-%d %H:%M:%S")
print( formatted_datetime)
visdom.text(f"start time {formatted_datetime}", win="starting time")

# Set seed
seed_value = 42
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
np.random.seed(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Custom Image Folder class
class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None,
                 image_folder="img", mask_folder="bbox"):
        super().__init__(root, transform=transform, target_transform=target_transform,
                         loader=loader, is_valid_file=is_valid_file)
        self.image_folder = image_folder
        self.mask_folder = mask_folder

    def __getitem__(self, index):
        file_path, _ = self.samples[index]
        filename = os.path.basename(file_path)
        image_path = os.path.join(self.root, self.image_folder, filename)
        mask_path = os.path.join(self.root, self.mask_folder, filename)

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            raise FileNotFoundError(f"Image or mask not found for file: {filename}")

        image = self.loader(image_path)
        mask = self.loader(mask_path)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

# Data module configuration
@dataclass
class ImageDataModuleConfig:
    data_dir: str
    test_data_dir: str
    image_size: Tuple[int, int]
    batch_size: int
    num_workers: int
    pin_memory: bool = True
    persistent_workers: bool = True

# Data module class
class ImageDataModule:
    def __init__(self, config: ImageDataModuleConfig) -> None:
        self.config = config
        self.train_dataset = None
        self.test_dataset = None
    
    def setup(self) -> None:
        transform = T.Compose([
            T.Resize(self.config.image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1),
        ])
        self.dataset = CustomImageFolder(
            self.config.data_dir,
            transform=transform,
            image_folder=os.path.join(args.data_dir, "img"),
            mask_folder=os.path.join(args.data_dir, "bbox")
        )
    
    def setup_test(self) -> None:
        transform = T.Compose([
            T.Resize(self.config.image_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: (x * 2) - 1),
        ])
        self.test_dataset = CustomImageFolder(
            self.config.test_data_dir,
            transform=transform,
            image_folder=os.path.join(args.test_data_dir, "img"),
            mask_folder=os.path.join(args.test_data_dir, "bbox")
        )

    def get_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            )
 
    def get_test_dataloader(self, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,)
 
# Training configuration
@dataclass
class TrainingConfig:
    lr: float = args.lr
    betas: Tuple[float, float] = (0.9, 0.995)
    sample_every_n_epochs = args.sample_every_n_epochs
    num_samples = args.num_samples
    sampling_sigmas: Tuple[Tuple[int, ...], ...] = (
        (80,),
        (80.0, 0.661),
        (80.0, 24.4, 5.84, 0.9, 0.661),
    )

# Function for sampling and logging samples
@torch.no_grad()
def sample_and_log_samples(batch: Union[Tensor, List[Tensor]], masks: Union[Tensor, List[Tensor]],
                            config: TrainingConfig, global_step) -> None:
    # Ensure the number of samples does not exceed the batch size
    num_samples = min(config.num_samples, batch.shape[0])

    # Log ground truth samples
    log_images(
        batch[:num_samples].detach(),
        "ground_truth",
        global_step,
        "ground_truth_window",
        window_size=(200, 200),
    )
    inpainting = ConsistencySamplingAndEditing()
    for sigmas in config.sampling_sigmas:
        samples = inpainting(
            model,
            batch.to(device),
            sigmas,
            masks,
            clip_denoised=True,
            verbose=True,
        )
        samples = samples.clamp(min=-1.0, max=1.0)

        log_images(
            samples,
            f"generated_samples-sigmas={sigmas}",
            global_step,
            f"generated_samples_window_{sigmas}",
            window_size=(400, 400),
        )

# Function for logging images
@torch.no_grad()
def log_images(images: Tensor, title: str, global_step: int, window_name: str, window_size: Tuple[int, int]) -> None:
    images = images.detach().float()

    grid = make_grid(
        images.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True, nrow=8
    )

    grid = grid.cpu().numpy()
    if grid.min() < 0:
        grid = (grid + 1) / 2
    width, height = window_size
    visdom.images(
        grid,
        nrow=4,
        opts=dict(
            title=f"{title} at step {global_step}",
            caption=f"{title}",
            width=width,
            height=height,
        ),
        win=window_name,
    )


def max_multiple(a,b):
    max_multiple = a // b
    max_number = max_multiple * b
    return max_number

def norm_tensor(tensor_ori):
    return (tensor_ori - tensor_ori.min())/(tensor_ori.max()-tensor_ori.min())



fid_metric = FrechetInceptionDistance()
lambda_gan = 0.9
# Main training loop
def train_one_epoch(epoch_index, epoch_length, device, config: TrainingConfig, train_loader, model,
                    optimizer_CT, visdom, netD, optimizer_D, test_loader):

    running_loss = 0.
    batchs_G_loss = 0.
    batchs_errD_loss = 0.
    batchs_loss_CT = 0.


    global_step = 0
    criterion = nn.BCELoss()
    real_label = 1.
    fake_label = 0.
    train_loader = tqdm(train_loader, desc=f'Epoch {epoch_index + 1}', dynamic_ncols=True)
    EPOCHS = int(args.max_steps / len(train_loader))

    max_batch_iter_idx = max_multiple(len(train_loader), args.iter_size)

    fake_image_list = []
    real_image_list = []
    loss_CT_list_iter = []
    loss_CT_list_batchs = []
    G_loss_list_iter = []
    G_loss_list_batchs = []
    errD_list_iter = []
    errD_list_batchs = []
    optimizer_CT.zero_grad()
    optimizer_D.zero_grad()
    for batch_idx, (images, masks) in enumerate(train_loader ):
        ################
        #(1)update Consistency training
        ################
                
        binary_masks = (masks != -1).float()
        consistency_training = ImprovedConsistencyTraining()
        global_step = epoch_index * epoch_length + batch_idx
        max_steps = EPOCHS * epoch_length

        output_CT = consistency_training(model, images.to(device), global_step, max_steps, binary_masks.to(device))
        fake = output_CT.predicted 
        loss_CT = (pseudo_huber_loss(fake, output_CT.target) * output_CT.loss_weights).mean()

        ###############
        #(1)update G network: maximinze log(D(G(z)))
        ################
        real_cpu = images.to(device)
        b_size = real_cpu.size(0)
        label_real_G = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        label_real_G.fill_(real_label)
        output_G = netD(fake, output_CT.next_sigmas).view(-1)  #####detach!!!!!!!
        errG = criterion(output_G, label_real_G)
        G_loss = loss_CT*(1-lambda_gan) + errG * lambda_gan

        if batch_idx < max_batch_iter_idx :
            G_loss_list_iter.append(G_loss.item())
            loss_CT_list_iter.append(loss_CT.item())
            G_loss.backward()
            batchs_G_loss += G_loss.item()
            batchs_loss_CT +=loss_CT.item()
            
            if (batch_idx+1) % args.iter_size == 0 :
                optimizer_CT.step()
                optimizer_CT.zero_grad()
                G_loss_list_batchs.append(batchs_G_loss / args.iter_size )
                loss_CT_list_batchs.append( batchs_loss_CT / args.iter_size )
                batchs_G_loss = 0
                batchs_loss_CT = 0

        else:
            G_loss_list_iter.append(G_loss.item())
            G_loss.backward()
            batchs_G_loss += G_loss.item()
            batchs_loss_CT +=loss_CT.item()
            
            if batch_idx == len(train_loader)-1:
                optimizer_CT.step()
                optimizer_CT.zero_grad()
                G_loss_list_batchs.append(batchs_G_loss / (len(train_loader) - max_batch_iter_idx) )
                loss_CT_list_batchs.append( batchs_loss_CT / ( len(train_loader) - max_batch_iter_idx ) ) 
                batchs_G_loss = 0
                batchs_loss_CT = 0

        ###############
        #(2)update D related network: maximinze log(D(x) + log(1-D(G(z)))
        ################
        
        #train with real batch
        label_real_D = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        label_real_D.fill_(real_label) 
        sigmas0 = torch.full((b_size,), 0.002, dtype=torch.float, device=device)
        output_real_D =  netD(real_cpu, sigmas0).view(-1)
        errD_real = criterion(output_real_D, label_real_D)
        
       #train with fake batch
        label_fake_D = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
        label_fake_D.fill_(fake_label)
        output_fake_D = netD(fake.detach(), output_CT.next_sigmas).view(-1)
        errD_fake = criterion(output_fake_D, label_fake_D) 


        errD = ( errD_real + errD_fake ) * lambda_gan
        if batch_idx < max_batch_iter_idx :
            errD_list_iter.append(errD.item())
            errD.backward()
            batchs_errD_loss += errD.item()
            
            if (batch_idx+1) % args.iter_size == 0 :
                optimizer_D.step()
                optimizer_D.zero_grad()
                errD_list_batchs.append(batchs_errD_loss / args.iter_size )
                batchs_errD_loss = 0

        else:
            errD_list_iter.append(errD.item())
            errD.backward()
            batchs_errD_loss += errD.item()
            
            if batch_idx == len(train_loader)-1:
                optimizer_D.step()
                optimizer_D.zero_grad()
                errD_list_batchs.append(batchs_errD_loss / (len(train_loader) - max_batch_iter_idx) )
                batchs_errD_loss = 0

        #
        # legend_labels = ["CT loss", "G loss", "D Loss"]
        # visdom.line(Y=[[loss_CT.detach().cpu().numpy(), errG.item(), errD.item()]],
        #         X=[global_step],
        #         win="loss_iteration", update="append",
        #         opts=dict(title="Three Losses over iteration", legend=legend_labels))
        #
         
        total_loss = G_loss + errD
        running_loss += total_loss.item()

        if batch_idx == len(train_loader) - 1:
            with open(f"loss_G_iter_itersize{args.iter_size}.txt", 'a') as file:
                file.write(f"{G_loss_list_iter}"+"\n")
                G_loss_list_iter=[]
            with open(f"loss_G_batchs_itersize{args.iter_size}.txt", 'a') as file:
                file.write(f"{G_loss_list_batchs}"+"\n")
                G_loss_list_batchs=[]
            with open(f"loss_CT_batchs_itersize{args.iter_size}.txt", 'a') as file:
                file.write(f"{loss_CT_list_batchs}"+"\n")
                loss_CT_list_batchs=[]
            with open(f"loss_CT_iter_itersize{args.iter_size}.txt", 'a') as file:
                file.write(f"{loss_CT_list_iter}"+"\n")
                loss_CT_list_iter=[]

            with open(f"errD_list_iter_itersize{args.iter_size}.txt", 'a') as file:
                file.write(f"{errD_list_iter}"+"\n")
                errD_list_iter=[]

            with open(f"errD_list_batchs_itersize{args.iter_size}.txt", 'a') as file:
                file.write(f"{errD_list_batchs}"+"\n")
                errD_list_batchs=[]

       
            inpainting_test = ConsistencySamplingAndEditing()
            sample_list = []
            real_list = []
            for batch_idx_test, (images_test, masks_test) in enumerate(test_loader):
                binary_masks_test = (masks_test != -1).float()
                sigmas = config.sampling_sigmas[-1]
                with torch.no_grad():
                    samples = inpainting_test( model,images_test.to(device),sigmas, binary_masks_test, clip_denoised=True,verbose=True)
                
                real_norm = norm_tensor(images_test)
                sample_norm = norm_tensor(samples)

                fid_metric.update(real_norm, is_real=True)
                fid_metric.update(sample_norm, is_real=False)
            
            fid_real_fake = fid_metric.compute()
            print(f"fid is {fid_real_fake}") 
            with open(f"fid_itersize{args.iter_size}.txt", 'a') as file:
                file.write(f"epoch {epoch_index}: {fid_real_fake}\n")
                visdom.line(Y=np.array([fid_real_fake.cpu().item()]),
                X=np.array([epoch_index]),
                win="fid over iteration for test dataset", update="append",
                opts=dict(title="FID iteration sampling"))

      
            sample_list = []
            real_list = []


        if (batch_idx) % len(train_loader) == 0:
            running_loss = 0.

        if batch_idx % 500 == 0:
            sample_and_log_samples(images.to(device), binary_masks.to(device), config, global_step)

    return running_loss / len(train_loader)

# Main training loop
def main():
    EPOCHS = int(args.max_steps / len(train_loader))
    global_step = 0

    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch + 1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch, len(train_loader), device, training_config, train_loader, model, optimizer_CT, visdom, netD, optimizer_D, test_loader)

        visdom.line([avg_loss], [epoch], win="consis_loss_epoch", opts=dict(title="loss over epoch time"), update="append")



        if epoch % args.sample_every_n_epochs == 0 or epoch == 0 or epoch == EPOCHS - 1:
            model_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}")
            model.save_pretrained(model_path)

            batch, masks = next(iter(train_loader))
            binary_masks = (masks != -1).float()
            sample_and_log_samples(batch.to(device), binary_masks.to(device), training_config, global_step)

if __name__ == "__main__":
    # Model configuration and summary
    model = UNet(UNetConfig()).to(device)
    optimizer_CT = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.995))

    netD = UNet_discriminator(UNetConfig_discriminator()).to(device)
    optimizer_D = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    # Training configuration
    training_config = TrainingConfig()

    # Data module
    dm_config = ImageDataModuleConfig(
        data_dir=args.data_dir,
        test_data_dir=args.test_data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    data_module = ImageDataModule(dm_config)
    data_module.setup()
    train_loader = data_module.get_dataloader(shuffle=True)
    print(f"train_loader len is {len(train_loader)}")

    test_dm_config = ImageDataModuleConfig(
        data_dir=args.data_dir,
        test_data_dir=args.test_data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    test_data_module = ImageDataModule(test_dm_config)
    test_data_module.setup_test()
    test_loader = test_data_module.get_test_dataloader(shuffle=True)
    print(f"test_loader len is {len(test_loader)}")


    # Extract the dataset name from the data_dir
    dataset_name = os.path.basename(args.data_dir)
    checkpoint_dir = os.path.join("checkpoints", dataset_name)
    print(f"checkpoint_dir is {checkpoint_dir}")

    # Create the dataset-specific checkpoint folder if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    visdom.text(f"lr_G is {args.lr},betas_G is (0.9, 0.995),lr_G is {args.lr}, betas_G is (0.5, 0.999), im_size is {args.image_size},  batch_size is {args.batch_size},max_steps is {args.max_steps}",
                win="text")
    main()
    end_time = datetime.now()
    formatted_datetime = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print( formatted_datetime)
    visdom.text(f"ending time {formatted_datetime}", win="ending time")


