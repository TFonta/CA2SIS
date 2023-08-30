## CA2-SIS: Semantic Image Synthesis with Class-adaptive Cross-attention

![image](./images/eyecatcher.png)
**Figure:** *Face images style and shape editing with CA2-SIS*

[[Paper](PUT-LINK)]

Semantic Image Synthesis with Class-adaptive Cross-attention (CA2-SIS) enables high reconstruction quality, style transfer from a reference image both at global (full) and class level, and automatic shape editing.

### Installation
Clone this repo:

```bash
git clone git@github.com:TFonta/CA2SIS.git
cd CA2SIS/

```

Install other requirements by running

```bash
pip install -r requirements.txt
```

### Datasets Preparation
To run the pre-trained models and generate reconstructed images or train a new model, the user needs to download the datasets and put them inside a 'datasets' folder in the root directory. Please refer also to this [link](https://github.com/ZPdesu/SEAN) for detailed instructions. 

Link to the pre-processed CelebAMask-HQ dataset: https://drive.google.com/file/d/1TKhN9kDvJEcpbIarwsd1_fsTR2vGx6LC/view

### Generating Images Using Pre-trained Models

To run the pre-trained models, please download the related weights and put the files into a 'checkpoints' folder in the root directory:

[[CelebMask-HQ (To be uploaded)](link)]
[[Ade20K (To be uploaded)](link)]

To test the pre-trained model run the following commands:

```bash
bash test_[dataset].sh
```
where `[dataset]` can either be `[celeba, ade20k]`.

### Train New Models

We trained our model on a single NVIDIA A100 GPU (40GB). The memory occupancy for training is non-negligible, so we recommend using a small batch size if training on GPUs with reduced memory.  

To train a new model, similarly to the testing case, you can run the following commands:

```bash
bash train_[dataset].sh
```
where `[dataset]` can either be `[celeba, ade20k]`.
 

### License
All rights reserved. Licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (**Attribution-NonCommercial-ShareAlike 4.0 International**) The code is released for academic research use only.

### Citation
If you use this code for your research, please cite our papers.

### Contacts
For any inquiries, feel free to contact tomaso.fontanini@unipr.it or ferrari.claudio88@gmail.com

### Acknowledgments
This code heavily borrows from the SEAN codebase https://github.com/ZPdesu/SEAN. 


