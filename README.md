# SAT
This is the revised version of SAT (Sememe Attention over Target) model, which is presented in the ACL 2017 paper **Improved Word Representation Learning with Sememes**. To get more details about the model, please read the [paper](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/acl2017_sememe.pdf) or access the [original project website](https://github.com/thunlp/SE-WRL).

## Updates
- Datasets：
	- Remove the wrong sememes `中国""Taiwan台湾` and `中国""Japan日本` from `SememeFile` and revise the corresponding lines in  `Word_Sense_Sememe_File`
	- Remove the single-sense words in `Word_Sense_Sememe_File`, which are not used in training process
- Input：
	- Learn vocabulary from the training file rather than read the existing vocabulary file.
- Output：
	- Output the vocabulary file learned from the training file
	- Output word, sense and sememe embeddings in 3 separate files
- Code: 
	- Rewrite most parts of the original code
	- Remove the redundant codes and rename some variables to improve readability.
	- Add evaluation programs including word similarity and analogy.
	- Add more comments

## How to Run

```
bash run_SAT.sh
```

To change training file, you can just switch the `data/train_sample.txt` in `run_SAT.sh` to your training file name.
## New Results

The results are based on the 21G `Sogou-T` as the training file, which can be downloaded from [here](https://pan.baidu.com/s/1kXgkyJ9) (password: f2ul). And the hyper-parameters for all the models are the same as those in `run_SAT.sh`. You can download the trained word embeddings from [here](https://cloud.tsinghua.edu.cn/d/76ab4a71efa541bd8eb3/).

### Word Similarity

|   Model   | Wordsim-240 | Wordsim-297 |
| :-------: | :---------: | :---------: |
|   CBOW    |    56.05    |    62.58    |
| Skip-gram |    56.72    |    61.99    |
|   GloVe   |    55.83    |    58.44    |
|    SAT    |  **62.11**  |  **62.74**  |

### Word Similarity


Model|city-acc|city-rank|family-acc|family-rank|capital-acc|capital-rank|total-acc|total-rank|
---|---|---|---|---|---|---|---|---|
Skip-gram|84.14|1.50|86.67|1.21|61.30|8.31|70.70|5.66|
SAT|98.85|1.01|77.20|5.27|80.06|10.10|82.29|7.52|

