# HTCInfoMax
The code for our NAACL 2021 paper "HTCInfoMax: A Global Model for Hierarchical Text Classification via Information Maximization".

## Requirements
+ Python >= 3.6
+ torch >= 0.4.1
+ numpy >= 1.17.4

## Preparation before train the model
### Data preprocess
#### dataset
+ Please get the original dataset of [RCV1-V2](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm) and [WoS](https://github.com/kk7nc/HDLTex)
+ use data.preprocess_rcv1_train.py and data.preprocess_rcv1_test.py to preprocess the RCV1-V2 dataset for hierarchical text classification.
+ use data.preprocess_wos.py to preprocess the WoS dataset for hierarchical text classification.

### Generate prior probability
+ run helper.hierarchy_tree_statistic_rcv1.py to generate the prior probability between parent-child pair of the label hierarchy in the training set of RCV1-V2.
+ run helper.hierarchy_tree_statistic_wos.py to generate the prior probability between parent-child pair of the label hierarchy in the training set of WoS.


## Train
To train the model on RCV1-V2 or WoS dataset, use the configuration file "htcinfomax-rcv1-v2.json" or "htcinfomax-wos.json" under "config" folder. Specifically, modify the line 162/163 in the train.py to use corresponding configuration file for the two datasets.

Then run the train.py file as follows:
```bash
python train.py
```

## Citation
If you find our paper or code is helpful for your work, please consider citing our NAACL 2021 paper, our paper is available at: https://www.aclweb.org/anthology/2021.naacl-main.260/
```bibtex
@inproceedings{deng-etal-2021-htcinfomax,
    title = "{HTCI}nfo{M}ax: A Global Model for Hierarchical Text Classification via Information Maximization",
    author = "Deng, Zhongfen  and
      Peng, Hao  and
      He, Dongxiao  and
      Li, Jianxin  and
      Yu, Philip",
    booktitle = "Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.naacl-main.260",
    doi = "10.18653/v1/2021.naacl-main.260",
    pages = "3259--3265",
}

```


## Acknowledgements
Our code is based on [HiAGM](https://github.com/Alibaba-NLP/HiAGM), we thank the authors of HiAGM for their open-source code.


