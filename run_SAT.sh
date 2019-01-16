make SAT_new
time ./SAT_new -train data/train_sample.txt -read-sememe data/SememeFile.txt -read-hownet data/Word_Sense_Sememe_File.txt -save-vocab vocab.txt -output-word word-vec.txt -output-sense sense-vec.txt -output-sememe sememe-vec.txt -size 200 -alpha 0.025 -min-count 50 -window 8 -sample 1e-4 -negative 25 -threads 20 -iter 3
python EvalWordSim.py word-vec.txt
python EvalAnalogy.py word-vec.txt