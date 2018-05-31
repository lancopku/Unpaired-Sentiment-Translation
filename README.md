# Unpaired-Sentiment-Translation
Code for "Unpaired Sentiment-to-Sentiment Translation: A Cycled Reinforcement Learning Approach" [[pdf]](https://arxiv.org/abs/1805.05181)
## Requirements
* ubuntu 16.04
* python 3.5
* tensorflow 1.4.1
* nltk 3.2.5
## Data
* [Yelp](https://www.yelp.com/dataset/challenge)
* [Amazon](https://snap.stanford.edu/data/web-FineFoods.html)
## Run
```bash
CUDA_VISIBLE_DEVICES=0 nohup python run_sentiment.py --mode=train --data_path=train/* --vocab_path=vocab.txt --log_root=log --exp_name=myexperiment --gpuid=0 > log.txt &
```
## Cite
To use this code, please cite the following paper:<br><br>
Jingjing Xu, Xu Sun, Qi Zeng, Xuancheng Ren, Xiaodong Zhang, Houfeng Wang, Wenjie Li.
Unpaired Sentiment-to-Sentiment Translation: A Cycled Reinforcement Learning Approach. In proceedings of ACL 2018.

bibtext:
```
@inproceedings{unpaired-sentiment-translation,
  author    = {Jingjing Xu and Xu Sun and Qi Zeng and Xuancheng Ren and Xiaodong Zhang and Houfeng Wang and Wenjie Li},
  title     = {Unpaired Sentiment-to-Sentiment Translation: A Cycled Reinforcement Learning Approach},
  booktitle = {ACL},
  year      = {2018}
}
```
