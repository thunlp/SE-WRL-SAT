import argparse
import json

import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wordvec')
    vecfile = parser.parse_args().wordvec
    wordvecdict = {}
    print('read word vectors...')
    with open(vecfile) as f:
        f.readline()
        num = 0
        for line in tqdm(f):
            num += 1
            items = line.strip().split()
            word = items[0]
            vec = np.array(list(map(float, items[1:])))
            wordvecdict[word] = vec
    print('word embeddings reading completes and total number of words is:', num)
    embedding = np.zeros((len(wordvecdict), wordvecdict['我'].size))
    word2index = dict()
    index2word = list()
    for word, vector in wordvecdict.items():
        index = len(word2index)
        word2index[word] = index
        embedding[index, :] = vector
        index2word.append(word)
    anadata = dict()
    rank_dict = dict()
    acc_dict = dict()
    category = None
    with open('./data/analogy.txt') as f:
        for line in f:
            if line.startswith(':'):
                category = line.strip().split()[1]
                anadata[category] = list()
                rank_dict[category] = list()
                acc_dict[category] = list()
                continue
            else:
                anadata[category].append(line.strip().split())
    total_valid_instance = 0
    total_instance = 0
    total_rank = 0
    total_acc = 0
    for category, instances in anadata.items():
        print(f'category {category}')
        for instance in tqdm(instances):
            if instance[0] not in word2index or instance[1] not in word2index or instance[2] not in word2index or instance[3] not in word2index:
                continue
            v1 = embedding[word2index[instance[1]]] - embedding[word2index[instance[0]]] + embedding[word2index[instance[2]]]
            target = word2index[instance[3]]
            distance2 = np.linalg.norm(embedding - v1, axis=1)
            top = np.where(distance2 < distance2[target])[0].tolist()
            rank = len(top) + 1
            if word2index[instance[0]] in top:
                rank -= 1
            if word2index[instance[1]] in top:
                rank -= 1
            if word2index[instance[2]] in top:
                rank -= 1
            if rank == 1:
                acc = 1
            else:
                acc = 0
            rank_dict[category].append(rank)
            acc_dict[category].append(acc)
        print(f'{category} acc = {sum(acc_dict[category]) / len(acc_dict[category])}, rank = {sum(rank_dict[category]) / len(acc_dict[category])}, coverage = {len(acc_dict[category]) / len(instances)}')
        total_instance += len(instances)
        total_valid_instance += len(rank_dict[category])
        total_rank += sum(rank_dict[category])
        total_acc += sum(acc_dict[category])
    print(f'total acc = {total_acc / total_valid_instance}, rank = {total_rank / total_valid_instance}, coverage = {total_valid_instance / total_instance}')
    json.dump(rank_dict, open('rank.json', 'w'))
