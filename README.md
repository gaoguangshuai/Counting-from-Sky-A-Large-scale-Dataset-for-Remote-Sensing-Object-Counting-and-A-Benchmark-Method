ASPDNet-pytorch
This is the PyTorch version for ASPDNet: "Counting from Sky£ºA Large-scale Dataset for Remote Sensing Object Counting and A Benchmark Method" in TGRS 2020, which delivered a state-of-the-art, straightforward and end-to-end architecture for object counting tasks.

***************************************************
Prerequisites
We strongly recommend Anaconda as the environment.

Python: 3.6

PyTorch: 1.0.1

CUDA: 10.0

***************************************************
Ground Truth
Please follow the make_dataset.py to generate the ground truth. It shall take some time to generate the dynamic ground truth. Note you need to generate your own json file.


***************************************************
Training Process
Try python train.py train.json val.json 0 0 to start training process.

***************************************************
Validation
Follow the val.py to try the validation.

***************************************************

Paper link: http://arxiv.org/abs/2008.12470

Dataset link：https://pan.baidu.com/s/19hL7O1sP_u2r9LNRsFSjdA  code：nwcx

***************************************************
References

If you find the ASPDNet useful, please cite our paper. Thank you!

@inproceedings{gao2020counting,
  title={Counting from Sky£ºA Large-scale Dataset for Remote Sensing Object Counting and A Benchmark Method},
  author={Gao, Guangshuai and Liu, Qingjie and Wang, Yunhong},
  booktitle={TGRS},
  year={2020}
}

@inproceedings{gao2020dense,
  title={Dense Object Counting in Remote Sensing Images},
  author={Gao, Guangshuai and Liu, Qingjie and Wang, Yunhong},
  booktitle={ICASSP},
  year={2020}
}


