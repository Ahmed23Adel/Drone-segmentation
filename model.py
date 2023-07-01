import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from colorama import Fore
import cv2
import os

COLORMAP_PATH = r'class_dict_seg.csv'

# Model Constants
CLASSES: int = 23
IN_CHANNELS = 3

# Set of models
deeplab_v3_plus_resnet34 = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_depth=5,
    encoder_weights="imagenet",
    decoder_channels=256,
    decoder_atrous_rates=(6, 12, 24),
    in_channels=IN_CHANNELS,
    classes=CLASSES,
    activation=None,
)

models = {
    "deeplab_v3_plus_resnet34": {
        "model": deeplab_v3_plus_resnet34, # Your model
        "optim_dict": None, # Optimization options
        "transforms_dict": None, # Transform Options
        "example_input_array": [1, 3, 704, 1056] # An example of the size of the input data on which to learn the neural network
    },
    # Add your dictionaries with models and their parameters as in the example
}


phase = 0
phases = len(models)
for key, value, in models.items():
    print(Fore.GREEN+f"\n|| Phase {phase}\n"
                      f"|| Model: {key}\n"+Fore.WHITE)

    model = value["model"]
    # Load the state_dict from the checkpoint
    checkpoint_path = r'epoch=29_val_loss=0.45_val_accuracy=0.86.ckpt'
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Create a new state_dict in which the keys don't contain the 'model.' prefix
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}

    # Load the state_dict into the model
    model.load_state_dict(state_dict)


weights: str = "/content/drive/MyDrive/self/Drone segmentation/lightning/epoch=29_val_loss=0.45_val_accuracy=0.86.ckpt"
# model = LightningPhotopicVisionModule.load_from_checkpoint(weights)

class SegmentationPostProcessing:
    def __init__(self,
                 model: nn.Module,
                 colormap_path: str,
                 device: str = None,
                 transforms = None,
                 height: int = 704,
                 width: int = 1056,
                 ):

        self.model = model
        self.__colormap = pd.read_csv(colormap_path)
        self.__colormap.pop("name")

        if not transforms:
            transforms = T.Compose([
                T.ToTensor(),
                T.Resize(size=(height, width)),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        self.transforms = transforms

        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device


    @staticmethod
    def __check_transforms(tensor_image: torch.Tensor) -> torch.Tensor:
        assert tensor_image.ndim == 4
        assert tensor_image.max() <= 3 and tensor_image.min() >= -3
        return tensor_image

    def __data_processing(self, image: np.array) -> torch.Tensor:
        tensor_image = self.transforms(image)
        tensor_image = tensor_image.to(device=self.device)
        tensor_image = tensor_image.unsqueeze(0)
        return self.__check_transforms(tensor_image)

    def __forward(self, x) -> torch.Tensor:
        self.model.to(device=self.device)
        self.model.eval()
        return self.model(x)

    def _predict_step(self, image: np.array) -> torch.Tensor:
        tensor_image = self.__data_processing(image=image)
        logites = self.__forward(tensor_image)
        activated = F.softmax(input=logites.squeeze(), dim=0)
        prediction = torch.argmax(input=activated, dim=0)
        return prediction.cpu()

    @staticmethod
    def _to_image(tensor_image: torch.Tensor) -> np.array:
        image = 255 * tensor_image.numpy().astype(np.uint8).transpose(1, 2, 0)
        return image

    @staticmethod
    def _to_array(tensor_image: torch.Tensor) -> np.array:
        image = tensor_image.numpy().astype(np.uint8)
        return image

    def _to_color(self, mask: np.array) -> np.array:
        mask = np.apply_along_axis(func1d=lambda index: self.__colormap.iloc[index],
                                   axis=1, arr=mask)
        return mask

    def _get_mask(self, image: np.array) -> np.array:
        tensor_mask = self._predict_step(image=image)
        mask = self._to_array(tensor_image=tensor_mask)
        mask = self._to_color(mask=mask)
        return mask

    def __call__(self, image: str) -> None:
        # image = cv2.imread(path, cv2.IMREAD_COLOR)
        mask = self._get_mask(image)


        image = cv2.resize(image, (mask.shape[1], mask.shape[0]), cv2.INTER_NEAREST)
        return image, mask


def create_dataset(data_path: str, target_path: str) -> pd.DataFrame:
    dict_paths = {
        "image": [],
        "mask": []
    }

    for dir_name, _, filenames in os.walk(data_path):
        for filename in filenames:
            name = filename.split('.')[0]
            dict_paths["image"].append(f"{data_path}/{name}.jpg")
            dict_paths["mask"].append(f"{target_path}/{name}.png")


    dataframe = pd.DataFrame(
        data=dict_paths,
        index=np.arange(0, len(dict_paths["image"]))
    )

    return dataframe


process = SegmentationPostProcessing(
    model=model,
    colormap_path=COLORMAP_PATH
)

def predict_img(img):
    numpy_img = np.asarray(img)
    _, mask = process(numpy_img)
    return mask

