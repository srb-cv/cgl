# Interpretable Concept groups on CNNs

## Introduction
This repository is based upon [NetDissect](https://github.com/CSAILVision/NetDissect), which contains the demo code for the work [Network Dissection: Quantifying Interpretability of Deep Visual Representations](http://netdissect.csail.mit.edu). 

The code is written in pytorch and python3.6, tested on Ubuntu 16.04. Please install [Pytorch](http://pytorch.org/) in python36 and [Torchvision](https://github.com/pytorch/vision) first to work with the code.


## Download
* Clone the code of Network Dissection Lite from github
```
    git clone https://github.com/srb-cv/cgl.git
```

## Training
In order to see all the flags to train the model:
```
python Places365/train_places_cnn.py --help
```

Example: Training an alexnet model on Places 365 model with different L-p norm regularizer, group activation loss, and spaltial loss:

```
python Places365/train_places_cnn.py /path-to-the-dataset -a alexnet --lr 0.01 --wd 0 --penalty 0.005 --activation-penalty 0 --spatial-penalty 0 --batch-size 256 --num_classes 365 --workers 7 --save /path-to-save-checkpoints --groups 5
```
Here, the loss function to train a model can be defined as :
$$L = L_d(w) + \lambda_{bn} R_{bn}(w) + \lambda_{g}L_g(\psi) + \lambda_{s}L_s(\psi)$$
where, 

$\lambda_{bn}$ for the **General Block Norm** can be set using the flag `--penalty`\
$\lambda_{g}$ for the **Group Activation Loss** can be set using the flag `--activation-penalty`\
$\lambda_{s}$ for the **Spatial Loss** can be set using the flag `--spatial-penalty`


## Evaluation
* Download the Broden dataset (~1GB space) and the example pretrained model. If you already download this, you can create a symbolic link to your original dataset.
```
    ./script/dlbroden.sh
    ./script/dlzoo_example.sh
```

* Run NetDissect
```
    python main.py
```


* At the end of the dissection script, a report will be generated inside `result` folder that summarizes the interpretable units of the tested network. These are, respectively, the HTML-formatted report, the semantics of the units of the layer summarized as a bar graph, visualizations of all the units of the layer (using zero-indexed unit numbers), and a CSV file containing raw scores of the top matching semantic concepts in each category for each unit of the layer.


## Reference
If you find the codes useful, please cite these papers
```
@inproceedings{ijcai2021-147,
  title     = {Learning Interpretable Concept Groups in CNNs},
  author    = {Varshneya, Saurabh and Ledent, Antoine and Vandermeulen, Robert A. and Lei, Yunwen and Enders, Matthias and Borth, Damian and Kloft, Marius},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {1061--1067},
  year      = {2021},
  month     = {8},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2021/147},
  url       = {https://doi.org/10.24963/ijcai.2021/147},
}

@inproceedings{netdissect2017,
  title={Network Dissection: Quantifying Interpretability of Deep Visual Representations},
  author={Bau, David and Zhou, Bolei and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
  booktitle={Computer Vision and Pattern Recognition},
  year={2017}
}

```
=======
# Learning Interpretable Concept Groups in CNNs
