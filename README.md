# This repo is built for paper: SEDNet_Shallow-Encoder-Decoder-Network-for-Brain-Tumor-Segmentation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/yourusername/yourproject.svg)](https://github.com/yourusername/yourproject/issues)
[![GitHub Stars](https://img.shields.io/github/stars/yourusername/yourproject.svg)](https://github.com/yourusername/yourproject/stargazers)

## Description

SEDNet is a segmentation algorithm that adopts sufficiency in hierarchical convolutional downsampling and selective skip mechanism for cost-efficient and effective Brain Tumor semantic segmentation. 
SEDNet with the integration of the proposed preprocessing algorithm and optimization function on the BraTS2020 set reserved for testing achieves impressive dice and Hausdorff scores of 0.9308, 0.9451,
0.9026, and 0.7040, 1.2866, 0.7762 for non-enhancing tumor core (NTC), peritumoral edema (ED), and enhancing tumor (ET), respectively. 
With about 1.3 million parameters and impressive performance in comparison to the state-of-the-art, SEDNet(X) is shown to be computationally efficient for real-time clinical diagnosis.  

<img src="SEDNet Architecture.png" width="900" height="200">

## Citation
If it is helpful for your work, please cite this paper:
```bash
@misc{olisah2024sednet,
      title={SEDNet: Shallow Encoder-Decoder Network for Brain Tumor Segmentation}, 
      author={Chollette C. Olisah},
      year={2024},
      eprint={2401.13403},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}

## Example Model Use
model_dir = "your model directory"
model = keras.models.load_model(model_dir, compile=False)
model.compile(loss=bce_dice_loss2, optimizer=keras.optimizers.Adam(learning_rate=0.0003), metrics = [dice_coef, sensitivity, specificity, C_0, C_1, C_2])

## License

This project is licensed under the MIT License.

Chollette, SEDNet_Shallow-Encoder-Decoder-Network-for-Brain-Tumor-Segmentation
