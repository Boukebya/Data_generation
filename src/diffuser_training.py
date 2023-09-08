# File containing methods to train the dataset. This file is used to train a diffuser model, it may not be used in
# the final product
from dataclasses import dataclass
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from diffusers import UNet2DModel
from diffusers import DDPMScheduler
from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from accelerate import Accelerator
from tqdm.auto import tqdm
import os


def make_grid(images, rows, cols):
    """
    This method makes a grid of images
    :param images: list of images to be gridded
    :param rows: Number of rows in the grid
    :param cols: Number of columns in the grid
    :return: Grid of images
    """
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline):
    """
    This method evaluates the model
    :param config:
    :param epoch:
    :param pipeline:
    :return:
    """
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    """
    This method trains the model
    :param config:
    :param model:
    :param noise_scheduler:
    :param optimizer:
    :param train_dataloader:
    :param lr_scheduler:
    :return:
    """
    # Initialize the accelerator which is responsible for distributing the model across
    accelerator = Accelerator(gradient_accumulation_steps=2)

    # Initialize a git repo and save the training code (empty bcs not using git)
    repo = []

    # Initialize the pipeline
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    # Now you train the model
    for epoch in range(config.num_epochs):
        # Initialize tqdm to track progress in the training loop
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        # Loop over all batches in an epoch
        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0].to(accelerator.device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,),
                                      device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Train the model on the noisy images
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                # Compute the MSE loss between the predicted noise and the true noise
                loss = F.mse_loss(noise_pred, noise)
                # Back-propagate the loss
                accelerator.backward(loss)

                # Update the model weights
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Update the progress bar
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            # accelerator.log(logs)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    # repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                    print("push to hub")
                else:
                    pipeline.save_pretrained(config.output_dir)


def train_stable(dataset_path):
    """
    This method trains the model with the stable dataset, need to implement an output and config file
    :return: None
    """

    # Define the training configuration and model hyperparameters
    @dataclass
    class TrainingConfig:
        image_size = 128  # the generated image resolution
        train_batch_size = 4  # how many images per batch
        eval_batch_size = 4  # how many images to sample during evaluation
        num_epochs = 1000  # how many times to go over the full dataset
        gradient_accumulation_steps = 1  # accumulate gradients over how many steps
        learning_rate = 1e-4  # the learning rate
        lr_warmup_steps = 200  # how many steps to warm up the learning rate
        save_image_epochs = 100  # save generated images every so many epochs
        save_model_epochs = 100  # save model checkpoint every so many epochs
        mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
        output_dir = 'Test'  # the model name locally and on the HF Hub

        push_to_hub = False  # whether to upload the saved model to the HF Hub
        hub_private_repo = False
        overwrite_output_dir = True  # overwrite the old model when re-running the notebook
        seed = 0

    config = TrainingConfig()


    # for each dataset pic, convert to tensor and resize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config.image_size, config.image_size)),
        transforms.Normalize([0.5], [0.5])
    ])

    # load the dataset
    dataset_path = torchvision.datasets.ImageFolder(dataset_path, transform=transform)

    # create a dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset_path, batch_size=config.train_batch_size, shuffle=True)

    train_dataloader = [(x[0], x[1]) for x in train_dataloader]

    # Define the model (UNet2d in this case)
    model = UNet2DModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet down-sampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet down-sampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ),
    )

    # sample img random img from dataset
    sample_image = train_dataloader[0][0]

    print('Input shape:', sample_image.shape)
    print('Output shape:', model(sample_image, timestep=0).sample.shape)

    # noise scheduler is used to add noise to the clean images during training
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # noise is a tensor of shape (batch_size, 3, image_size, image_size), it's added to the clean images
    noise = torch.randn(sample_image.shape)

    # timesteps is a tensor of shape (batch_size, 1), it's used to schedule the noise level
    timesteps = torch.LongTensor([50])
    # we add noise to the sample image
    noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

    # convert the noisy image to a PIL image
    Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

    # show noisy image
    plt.imshow(Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0]))
    plt.show()

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # Define the learning rate scheduler which will be used to adjust the learning rate during training
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    # args for train loop
    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

    train_loop(*args)
