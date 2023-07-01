# Drone-segmentation

# 0.0 Introduction

Drone Aerial Segmentation project

<img src="https://media.tenor.com/RGzSNBinFcEAAAAS/drone.gif" width="500">

The Semantic Drone Dataset is a valuable resource for advancing the field of autonomous drone flight and enhancing the safety of urban operations. With comprehensive annotations for person detection and semantic segmentation, this dataset enables the development and evaluation of algorithms to accurately detect persons and classify various urban elements like trees, grass, buildings, and obstacles. These applications have significant implications for enhancing the safety and efficiency of autonomous drones in urban environments.

# Output:

<img width="914" alt="Screenshot 2023-07-01 at 10 33 30 PM" src="https://github.com/Ahmed23Adel/Drone-segmentation/assets/69484554/0e9e5c5f-d34f-4d69-aac7-cefab0cad3b1">

<img width="930" alt="Screenshot 2023-07-01 at 10 34 07 PM" src="https://github.com/Ahmed23Adel/Drone-segmentation/assets/69484554/e86e2f10-a8f4-4911-be7c-4d819b805208">

<img width="913" alt="Screenshot 2023-07-01 at 10 34 52 PM" src="https://github.com/Ahmed23Adel/Drone-segmentation/assets/69484554/d0934a91-c74d-4c57-aa88-133dba648aa1">

# Deployement:
I used streamlit, it's easy and fast-to-learn library and match project that includes data science


<img width="1280" alt="Screenshot 2023-07-01 at 10 35 41 PM" src="https://github.com/Ahmed23Adel/Drone-segmentation/assets/69484554/389b1260-d17d-4109-b64b-c6c3f5e50982">

<img width="832" alt="Screenshot 2023-07-01 at 10 36 01 PM" src="https://github.com/Ahmed23Adel/Drone-segmentation/assets/69484554/9c112a39-bded-4ee1-bc3e-1b5ceedd4b8f">

<img width="1249" alt="Screenshot 2023-07-01 at 10 36 29 PM" src="https://github.com/Ahmed23Adel/Drone-segmentation/assets/69484554/c254910c-3245-4421-8cd7-6e19b6140ecb">

<img width="906" alt="Screenshot 2023-07-01 at 10 36 42 PM" src="https://github.com/Ahmed23Adel/Drone-segmentation/assets/69484554/2ba932f9-133f-4805-8669-ea711df0d328">


# Details:
I used PyTorch lightning for this project

Model details:
We used DeepLabV3Plus and resnet34 for feature extraction,
 and for optimization I used Adam, and cross entropy loss.

DeepLabV3Plus:

![model_arch](https://github.com/Ahmed23Adel/Drone-segmentation/assets/69484554/716b5b49-f066-416e-aa3e-6d5f787e33d4)


I used albumentations for transforms, it generates sunny, rainy effect and many more

For metric i used Accurate, jaccard index(IOU)



