//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define EXP_FROM_ZERO_FORE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000


const int vocab_hash_size = 10000000;
const int sememe_hash_size = 10000000;

typedef float real;

struct vocab_word {
  long long cn;
  char *word;
};

struct sememe_word {
  char *word;
};

struct sense {
  int num;
  real *sense_vecs;
  int *sememe_num_acc;
  int *sememe_idx;
};


char train_file[MAX_STRING], word_vec_file[MAX_STRING];

char save_vocab_file[MAX_STRING], read_sememe_file[MAX_STRING], read_hownet_file[MAX_STRING];
char sense_vec_file[MAX_STRING], sememe_vec_file[MAX_STRING];

struct vocab_word *vocab;
struct sense *word_senses;
struct sememe_word *sememes;

int debug_mode = 2, window = 5, min_count = 5, num_threads = 12, min_reduce = 1;

int *vocab_hash, *sememe_hash;

long long vocab_max_size = 1000, vocab_size = 0, vec_dim = 100;
long long train_words = 0, word_count_actual = 0, iter = 5, file_size = 0;

long long sememe_size = 0;


real alpha = 0.025, starting_alpha, sample = 1e-3;
real *word_vec, *syn1, *syn1neg, *expTable ;

real *pre_exp;
real *sememe_vec;

clock_t start;

int negative = 5;
const int table_size = 1e8;
int *table;

// Read an integer from the word_sense_semem file
int ReadInteger(FILE *fin) {
  int a = 0, ch, sub = '0';
  while (!feof(fin)) {
    ch = fgetc(fin);
    if ((ch == ' ') || (ch == '\n'))
      break;
    a = a * 10 + ch - sub;
  }
  return a;
}

// Read a word from the word_sense_semem file
void ReadVocabWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if ((ch == ' ') || (ch == '\n'))
      break;
    word[a] = ch;
    ++a;
    if (a >= MAX_STRING - 1)
      --a;
  }
  word[a] = 0;
}


// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}
// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Returns position of a word in the sememes vocabulary; if the word is not found, returns -1
int SearchSememe(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (sememe_hash[hash] == -1) return -1;
    if (!strcmp(word, sememes[sememe_hash[hash]].word)) return sememe_hash[hash];
    hash = (hash + 1) % sememe_hash_size;
  }
  return -1;
}

// Read the word_sense_semem file
void ReadHowNet() {
  FILE *fin = fopen(read_hownet_file, "rb");
  if (fin == NULL) {
    printf("HowNet file not found\n");
    exit(1);
  }

  char word[MAX_STRING];
  int sememe_ids[111], cnt[111], i, j;
  unsigned long long next_random = 1;
  long long a;

  a = posix_memalign((void **)&word_senses, 128, (long long)vocab_size * sizeof(struct sense));
  if (word_senses == NULL) {printf("Memory allocation failed\n"); exit(1);}

  for (i = 0; i < vocab_size; i++) {
    word_senses[i].num = 1;
    word_senses[i].sense_vecs = (real *)calloc(1 * vec_dim, sizeof(real));
    for (j = 0; j < vec_dim; j++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      word_senses[i].sense_vecs[j] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / vec_dim;
    }
  }


  while (1) {

    ReadVocabWord(word, fin);
    if (feof(fin)) break;

    int word_id = SearchVocab(word);
    int sense_num = ReadInteger(fin);

    if (sense_num > 1) {
      for (i = 0; i < sense_num; ++i) {
        int sememe_cnt = ReadInteger(fin);
        int sememe_id_tmp;

        cnt[i] = (i == 0 ? 0 : cnt[i - 1]) + sememe_cnt; // the accumulated number of sememes
        for (j = (i == 0 ? 0 : cnt[i - 1]); j < cnt[i]; ++j) {
          ReadVocabWord(word, fin);
          sememe_id_tmp = SearchSememe(word);
          if (sememe_id_tmp == -1) { // For unknown sememes
            cnt[i]--;
            j--;
          }
          else {
            sememe_ids[j] = sememe_id_tmp;
          }
        }
      }
    }
    if (word_id != -1) {
      word_senses[word_id].num = sense_num;

      if (sense_num > 1) {

        word_senses[word_id].sense_vecs = (real *)realloc(word_senses[word_id].sense_vecs, sense_num * vec_dim * sizeof(real));
        for (i = 0; i < sense_num * vec_dim; ++i) {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          word_senses[word_id].sense_vecs[i] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / vec_dim;
        }

        word_senses[word_id].sememe_num_acc = (int *)calloc(sense_num, sizeof(int));
        for (i = 0; i < sense_num; ++i)
          word_senses[word_id].sememe_num_acc[i] = cnt[i];

        word_senses[word_id].sememe_idx = (int *)calloc(word_senses[word_id].sememe_num_acc[sense_num - 1], sizeof(int));
        for (i = 0; i < word_senses[word_id].sememe_num_acc[sense_num - 1]; ++i)
          word_senses[word_id].sememe_idx[i] = sememe_ids[i];
      }
    }
  }
  printf("HowNet Read End\n");
  fflush(stdout);
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}


// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;

  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;

  return vocab_size - 1;
}


// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
  return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 0; a < size; a++) {
    // Words occuring less than min_count times will be discarded from the vocab
    if ((vocab[a].cn < min_count) && (a != 0)) {
      vocab_size--;
      free(vocab[a].word);
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash = GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));

}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++)
    if (vocab[a].cn > min_reduce) {
      memcpy(vocab + b, vocab + a, sizeof(struct vocab_word));
      b++;
    } else {
      free(vocab[a].word);
    }
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Learn vocabulary from training file
void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");

  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }

  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
    fflush(stdout);
  }
  file_size = ftell(fin);
  fclose(fin);
}


// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadSememeWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if ((ch == ' ') || (ch == '\n'))
      break;
    word[a] = ch;
    ++a;
    if (a >= MAX_STRING - 1)
      --a;
  }
  word[a] = 0;
}

// Adds a sememes to the sememes vocabulary
int AddWordToSememeVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  sememes[sememe_size].word = (char *)calloc(length, sizeof(char));
  strcpy(sememes[sememe_size].word, word);
  sememe_size++;
  hash = GetWordHash(word);
  while (sememe_hash[hash] != -1) hash = (hash + 1) % sememe_hash_size;
  sememe_hash[hash] = sememe_size - 1;
  return sememe_size - 1;
}

// Read all the sememes from the SememeFile
void ReadSememe() {
  long long a;
  char word[MAX_STRING];
  FILE *fin = fopen(read_sememe_file, "rb");

  if (fin == NULL) {
    printf("Sememe Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < sememe_hash_size; a++) sememe_hash[a] = -1;

  sememe_size = 0;
  while (1) {
    ReadSememeWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToSememeVocab(word);
  }
  fclose(fin);
  printf("Sememe Number: %lld\n", sememe_size);
  fflush(stdout);
}

// Initialize the vectors
void InitNet() {
  long long a, b;
  unsigned long long next_random = 1;

  a = posix_memalign((void **)&word_vec, 128, (long long)vocab_size * vec_dim * sizeof(real));
  if (word_vec == NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (a = 0; a < vocab_size; a++) for (b = 0; b < vec_dim; b++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      word_vec[a * vec_dim + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / vec_dim;
    }

  a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * vec_dim * sizeof(real));
  if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
  for (a = 0; a < vocab_size; a++) for (b = 0; b < vec_dim; b++)
      syn1neg[a * vec_dim + b] = 0;

  a = posix_memalign((void **)&sememe_vec, 128, (long long)sememe_size * vec_dim * sizeof(real));
  if (sememe_vec == NULL) {printf("Memory allocation failed\n"); exit(1);}

  for (a = 0; a < sememe_size; a++) for (b = 0; b < vec_dim; b++) {
      next_random = next_random * (unsigned long long)25214903917 + 11;
      sememe_vec[a * vec_dim + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / vec_dim;
    }
}

// Do dot product for two vectors
real vectorDot(real *a, real *b, int dim) {
  int i;
  real dot = 0;
  for (i = 0; i < dim; ++i)
    dot += a[i] * b[i];
  return dot;
}

// Main training  function
void *TrainModelThread(void *id) {
  // Common Part
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;

  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);

  // real *neu1 = (real *)calloc(vec_dim, sizeof(real));
  real *neu1e = (real *)calloc(vec_dim, sizeof(real));

  // Sememe Part
  int p, q;

  real *attention = (real *)calloc(vec_dim, sizeof(real));
  real *context_vec = (real *)calloc(vec_dim, sizeof(real));
  real *attention_avg = (real *)calloc(vec_dim, sizeof(real));
  real *_exp = (real *)calloc(223, sizeof(real));
  real *mult_part = (real *)calloc(vec_dim, sizeof(real));
  real *mult_part2 = (real *)calloc(vec_dim, sizeof(real));
  real *avg_sememe = (real *)calloc(vec_dim * 35, sizeof(real));
  real total = 0;


  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now = clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
               word_count_actual / (real)(iter * train_words + 1) * 100,
               word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
        // if (word_count_actual / (real)(iter * train_words + 1) * 100 > 5) break;
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) {
          break;
        }
        if (word == -1) {
          continue;
        }
        word_count++;

        if (word == 0) break;

        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi) || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0) break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }

    word = sen[sentence_position];
    if (word == -1) continue;

    //calc the average of context word embeddings

    for (a = 0; a < vec_dim; ++a) {
      context_vec[a] = 0;
      neu1e[a] = 0;
    }
    real cnt = 0;

    int window2 = 2;
    for (a = 0; a < window2 * 2 + 1; a++)
      if (a != window2) {
        c = sentence_position - window2 + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;

        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * vec_dim;

        cnt += 1.0;
        for (c = 0; c < vec_dim; ++c)
          context_vec[c] += word_vec[l1 + c];
      }

    if (cnt > 0) {
      for (a = 0; a < vec_dim; ++a)
        context_vec[a] /= cnt;
      cnt = 1.0 / cnt;
    }

    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;

    for (a = b; a < window * 2 + 1 - b; a++)
      if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;

        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * vec_dim;

        // NEGATIVE SAMPLING
        if (negative > 0)
          for (d = 0; d < negative + 1; d++) {
            if (d == 0) {
              target = word;
              label = 1;
            } else {
              next_random = next_random * (unsigned long long)25214903917 + 11;
              target = table[(next_random >> 16) % table_size];
              if (target == 0) target = next_random % (vocab_size - 1) + 1;
              if (target == word) continue;
              label = 0;
            }
            f = 0;
            if (word_senses[target].num == 1) {
              total = 1;
              for (p = 0; p < vec_dim; ++p) attention[p] = word_senses[target].sense_vecs[p];
            }
            else {
              real temp_total = 0;
              for (p = 0; p < vec_dim; ++p) attention[p] = 0;
              for (p = 0; p < vec_dim; ++p) attention_avg[p] = 0;
              real min_exp = 10000000000000;
              for (p = 0; p < word_senses[target].num; ++p) {
                // calc the average of sememe embeddings for each sense
                for (q = 0; q < vec_dim; ++q)
                  avg_sememe[p * vec_dim + q] = 0;
                for (q = (p == 0 ? 0 : word_senses[target].sememe_num_acc[p - 1]); q < word_senses[target].sememe_num_acc[p]; ++q) {
                  real *temp = &(sememe_vec[word_senses[target].sememe_idx[q] * vec_dim]);
                  for (c = 0; c < vec_dim; ++c) {
                    avg_sememe[p * vec_dim + c] += temp[c];
                  }
                }
                real divide_part;
                if (p == 0) {
                  divide_part = 1.0 / (real)(word_senses[target].sememe_num_acc[0]);
                }
                else {
                  divide_part = 1.0 / (real)(word_senses[target].sememe_num_acc[p] - word_senses[target].sememe_num_acc[p - 1]);
                }
                for (c = 0; c < vec_dim; ++c)
                  avg_sememe[p * vec_dim + c] *= divide_part;
                // calc the weight for each sense
                _exp[p] = vectorDot(&(avg_sememe[p * vec_dim]), context_vec, vec_dim);
                if (_exp[p] < min_exp)
                  min_exp = _exp[p];
              }
              for (p = 0; p < word_senses[target].num; ++p) {
                _exp[p] = _exp[p] - min_exp;
                if (_exp[p] >= 3.999)
                  _exp[p] = 3.999;
                _exp[p] = pre_exp[(int)(_exp[p] * 250.0)];
                temp_total += _exp[p];
              }
              // get "attention" result on senses
              for (p = 0; p < word_senses[target].num; ++p) {
                _exp[p] = _exp[p] / temp_total;
                for (q = 0; q < vec_dim; ++q) {
                  attention[q] += _exp[p] * word_senses[target].sense_vecs[p * vec_dim + q];
                  attention_avg[q] += _exp[p] * avg_sememe[p * vec_dim + q];
                }
              }
              total = temp_total;
            }

            // BP
            for (c = 0; c < vec_dim; c++) {
              f += word_vec[c + l1] * attention[c];
            }
            if (f > MAX_EXP)
              g = (label - 1) * alpha;
            else if (f < -MAX_EXP)
              g = (label - 0) * alpha;
            else
              g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
            g /= total;
            for (c = 0; c < vec_dim; c++) {
              neu1e[c] = g * attention[c];
            }
            if (word_senses[target].num == 1) {
              for (p = 0; p < vec_dim; ++p)
                word_senses[target].sense_vecs[p] += g * word_vec[p + l1];
            }
            else {
              for (p = 0; p < vec_dim; ++p)
                mult_part[p] = g * word_vec[p + l1] * context_vec[p];
              real avg_num;
              for (p = 0; p < word_senses[target].num; ++p) {
                if (p == 0)
                  avg_num = 1.0 / (real)(word_senses[target].sememe_num_acc[0]);
                else
                  avg_num = 1.0 / (real)(word_senses[target].sememe_num_acc[p] - word_senses[target].sememe_num_acc[p - 1]);
                real *temp_mult = &(word_senses[target].sense_vecs[p * vec_dim]);
                for (c = 0; c < vec_dim; ++c) {
                  mult_part2[c] = mult_part[c] * (temp_mult[c] - attention[c]) * _exp[p] * avg_num;
                }
                for (q = (p == 0 ? 0 : word_senses[target].sememe_num_acc[p - 1]); q < word_senses[target].sememe_num_acc[p]; ++q) {
                  real *temp = &(sememe_vec[word_senses[target].sememe_idx[q] * vec_dim]);
                  for (c = 0; c < vec_dim; ++c) {
                    temp[c] += mult_part2[c];
                  }
                }
              }

              for (p = 0; p < vec_dim; ++p)
                mult_part[p] = 0;
              for (p = 0; p < word_senses[target].num; ++p) {
                real *temp_mult = &(word_senses[target].sense_vecs[p * vec_dim]);
                for (c = 0; c < vec_dim; ++c)
                  mult_part[c] += _exp[p] * temp_mult[c] * (avg_sememe[p * vec_dim + c] - attention_avg[c]);
              }
              for (p = 0; p < vec_dim; ++p)
                mult_part[p] *= g * word_vec[p + l1] * cnt;
              for (q = 0; q < window2 * 2 + 1; q++) if (q != window2) {
                  p = sentence_position - window2 + q;
                  if (p < 0) continue;
                  if (p >= sentence_length) continue;
                  p = sen[p];
                  if (p == -1) continue;
                  p = p * vec_dim;
                  for (c = 0; c < vec_dim; ++c)
                    word_vec[p + c] += mult_part[c];
                }

              for (p = 0; p < vec_dim; ++p)
                mult_part[p] = g * word_vec[p + l1];
              for (p = 0; p < word_senses[target].num; ++p) {
                real *temp_mult = &(word_senses[target].sense_vecs[p * vec_dim]);
                for (q = 0; q < vec_dim; ++q)
                  temp_mult[q] += _exp[p] * mult_part[q];
              }
            }

            for (c = 0; c < vec_dim; c++) {
              word_vec[c + l1] += neu1e[c];
            }
          } // Eng of Negative Sampling
      } // End of Current word handling for SE-WRL
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  } // End of While

  fclose(fi);

  free(neu1e);
  // free(neu1);

  free(attention);
  free(context_vec);
  free(attention_avg);
  free(_exp);
  free(mult_part);
  free(mult_part2);
  free(avg_sememe);


  printf("train end\n");
  fflush(stdout);
  pthread_exit(NULL);
}

void InitUnigramTable() {
  int a, i;
  double train_words_pow = 0;
  double d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (double)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}


void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
  printf("Vocabulary Saved\n");
  fflush(stdout);
}


void TrainModel() {
  long a, b;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;

  LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (read_sememe_file[0] != 0) ReadSememe();
  if (read_hownet_file[0] != 0) ReadHowNet();

  if (word_vec_file[0] == 0) return;

  InitNet();

  if (negative > 0) InitUnigramTable();

  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);



  printf("all trains end\n");
  printf("%lld %lld %lld\n", sememe_size, vocab_size, vec_dim);

  // Save the word vectors
  if (word_vec_file[0] != 0) {
    fo = fopen(word_vec_file, "w");
    fprintf(fo, "%lld %lld\n", vocab_size, vec_dim);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      for (b = 0; b < vec_dim; b++)
        fprintf(fo, "%lf ", word_vec[a * vec_dim + b]);
      fprintf(fo, "\n");
    }
    fclose(fo);
  }

  // Save the sense vectors
  if (sense_vec_file[0] != 0) {
    fo = fopen(sense_vec_file, "w");
    fprintf(fo, "%lld %lld\n", vocab_size, vec_dim);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", vocab[a].word);
      fprintf(fo, "%d ", word_senses[a].num);
      for (b = 0; b < word_senses[a].num * vec_dim; b++)
        fprintf(fo, "%lf ", word_senses[a].sense_vecs[b]);
      if (word_senses[a].num > 1) {
        for (b = 0; b < word_senses[a].num; ++b)
          fprintf(fo, "%d ", word_senses[a].sememe_num_acc[b]);
        for (b = 0; b < word_senses[a].sememe_num_acc[word_senses[a].num - 1]; ++b)
          fprintf(fo, "%d ", word_senses[a].sememe_idx[b]);
      }
      fprintf(fo, "\n");
    }
    fclose(fo);
  }

  // Save the Sememe vectors
  if (sememe_vec_file[0] != 0) {
    fo = fopen(sememe_vec_file, "w");
    fprintf(fo, "%lld %lld\n", sememe_size, vec_dim);
    for (a = 0; a < sememe_size; a++) {
      fprintf(fo, "%s ", sememes[a].word);
      for (b = 0; b < vec_dim; b++)
        fprintf(fo, "%lf ", sememe_vec[a * vec_dim + b]);
      fprintf(fo, "\n");
    }
    fclose(fo);
  }

  printf("save end\n");
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
    }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");

    printf("\t-output-word <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-output-sense <file>\n");
    printf("\t\tUse <file> to save the resulting sense vectors\n");
    printf("\t-output-sememe <file>\n");
    printf("\t\tUse <file> to save the resulting sememe vectors\n");

    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
    printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");

    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 12)\n");
    printf("\t-iter <int>\n");
    printf("\t\tRun more training iterations (default 5)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    return 0;
  }

  read_sememe_file[0] = 0;
  read_hownet_file[0] = 0;


  save_vocab_file[0] = 0;
  word_vec_file[0] = 0;

  sense_vec_file[0] = 0;
  sememe_vec_file[0] = 0;


  // Input
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);

  if ((i = ArgPos((char *)"-read-sememe", argc, argv)) > 0) strcpy(read_sememe_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-hownet", argc, argv)) > 0) strcpy(read_hownet_file, argv[i + 1]);

  // Output
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-output-word", argc, argv)) > 0) strcpy(word_vec_file, argv[i + 1]);

  if ((i = ArgPos((char *)"-output-sense", argc, argv)) > 0) strcpy(sense_vec_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-output-sememe", argc, argv)) > 0) strcpy(sememe_vec_file, argv[i + 1]);

  // word2vec
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) vec_dim = atoi(argv[i + 1]);

  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);

  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]); iter = 3;

  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

  sememes = (struct sememe_word *)calloc(2000, sizeof(struct sememe_word));
  sememe_hash = (int *)calloc(sememe_hash_size, sizeof(int));


  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }

  pre_exp = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_FROM_ZERO_FORE; ++i)
    pre_exp[i] = exp((real)i / (real)(EXP_FROM_ZERO_FORE / 4));

  TrainModel();
  return 0;
}
