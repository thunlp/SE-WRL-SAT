# SAT
This is the revised version of SAT (Sememe Attention over Target) model, which is presented in the ACL 2017 paper **Improved Word Representation Learning with Sememes**. To get more details about the model, please read the [paper](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/acl2017_sememe.pdf) or access the [original project website](http://https://github.com/thunlp/SE-WRL).

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
	- Add more comments

## How to Run

```
bash run_SAT.sh
```

To change training file, you can just switch the `data/train_sample.txt` in `run_SAT.sh` to your training file name.
## New Results

The results are based on the 21G `Sogou-T` as the training file. And the hyper-parameters for all the models are the same as those in `run_SAT.sh`.

### Word Similarity

|   Model   | Wordsim-240 | Wordsim-297 |
| :-------: | :---------: | :---------: |
|   CBOW    |    56.05    |    62.58    |
| Skip-gram |    56.72    |    61.99    |
|   GloVe   |    55.83    |    58.44    |
|    SAT    |  **61.00**  |  **64.38**  |
