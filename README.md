# JLM
A fast LSTM Language Model for large vocabulary language like Japanese and Chinese.

It focuses on **accelerating inference time** and **reducing model size** to fit requirement of real-time applications especially in client side. With [BCCWJ](http://pj.ninjal.ac.jp/corpus_center/bccwj/en/) Japanese corpus, it is **85% smaller**, and are **50x faster** than standard LSTM solution with softmax. See the paper (coming soon at ANLP 2018) for performance detail.

The training part is done with TensorFlow. We then dumped the trained weights out. The inference and decoding are done with numpy for example here. You can play C++ with Eigen as a very easy alternative.

## Language model
We implemented the standard LSTM , [tie-embedding](https://arxiv.org/abs/1608.05859), [D-softmax](https://arxiv.org/abs/1512.04906), and [D-softmax*](https://arxiv.org/abs/1609.04309) in both training and numpy inference stages as a comparison. In practice, please consider just use D-softmax* for your best interest.

## Decoder
For Chinese and Japanese input, decoder is used to decode the converted sentence from user keyboard input. It is also useful in many other applications like the LM stage of speech recognition, or the output part of seq2seq model.

We implemented a standard Viterbi decoder with beam search. A very important trick to combining decoder with deep learning LM is batching. Batch all the queries in a path all together saves tons of time.

# How to use
## Corpus preparation
Make sure you have a txt file with a white space segmented sentence per line.
> If not, use tools like [MeCab](http://taku910.github.io/mecab/) to get the job done first.

Chinese and Japanese have words with different pronunciations. It is better to tell them as different words. In JLM, the unit **word** is the in the format of **display/reading/POS**.  For example "品川/シナガワ/名詞-固有名詞-地名-一般".

## Run data preparation script
> python data.py -v 50000

Make sure you put your corpus.txt in the data folder first. The script will generate a encoded corpus and a bunch of pickles for later usage.  The lexicon is also generated in this stage. Words in lexicon are ordered by their frequency for the convenience of vocabulary segmentation.

## Run training script
- Config the experiment folder path (put the corpus with vocab you would like to use in the config)
> data_path = os.path.abspath(os.path.join(root_path, "data/corpus_50000"))

- Enter the train folder, and edit the run file [train.py](https://github.com/jiali-ms/JLM/blob/master/train/train.py) file

```python
parameters = {
    "is_debug": False,
    "batch_size": 128 * 3,
    "embed_size": 256,
    "hidden_size": 512,
    "num_steps": 25,
    "max_epochs": 10,
    "early_stopping": 1,
    "dropout": 0.9,
    "lr": 0.001,
    "share_embedding": True,
    "gpu_id": 0,
    "tf_random_seed": 101,
    "D_softmax": False,
    "V_table": True,
    "embedding_seg": [(200, 0, 12000), (100, 12000, 30000), (50, 30000, None)],
    "tune": False,
    "tune_id": -1,
    "tune_lock_input_embedding": False
}
```
 **V_table** is the D-softmax* here. The **embedding_seg** means how you want to separate your vocabulary. Make your own best result by tuning the hyper parameters. The results will be saved to folder "experiments" with a ID, **sacred** framework will take care of all the experiment indexing.
 > A small hint here, use up your GPU memory by setting a bigger **num_steps** and **batch_size** to best reduce training time.

## Verify LM is correct
Run the [test.py](https://github.com/jiali-ms/JLM/blob/master/train/test.py) in train folder, it will auto generate a random sentence with the trained model. If the sentence doesn't make sense to your language knowledge at all. One of the stage is not correctly setup. Here is an example of random generated sentence.
> 自宅 に テレビ 局 で 電話 し たら 、 で も 、 すべて の 通話 が 実現 。

## Dump  the TF trained weights
It is a common issue to bring trained model into clients like mobile device or Windows environment. We recommend you use ONNX for the purpose. But for this example, we want **full** control and want to cutoff dependency to TF. Run the [weights.py](https://github.com/jiali-ms/JLM/blob/master/train/weights.py) in the train folder to get a pickle of all the trainable parameters in the numpy format. You can use the following command to even dump the txt format for other usage.
> weiths.py -e 1 -v True
## Offline model and decoder
Finally, we have trained weights, just need to load them with a offline model! In our architecture, the offline model can be anything you like, for simplify, we choose python and numpy. [model.py](https://github.com/jiali-ms/JLM/blob/master/decoder/model.py) is the python version of the **exact** model we used in training.

The [decoder.py](https://github.com/jiali-ms/JLM/blob/master/decoder/decoder.py) implements a standard Viterbi with beam search. The predictions from the same frame are batched to accelerate the  matrix op. Run the script to see an example decoded result from converting "キョーワイーテンキデス".
> python decoder.py


`(17.569383530169105, ['今日/キョー/名詞-普通名詞-副詞可能', 'は/ワ/助詞-係助詞', 'いい/イー/形容詞-非自立可能', '天気/テンキ/名詞-普通名詞-一般', 'です/デス/助動詞'])`

`(27.574523992770953, ['鏡/キョー/接尾辞-名詞的-一般', 'は/ワ/助詞-係助詞', 'いい/イー/形容詞-非自立可能', '天気/テンキ/名詞-普通名詞-一般', 'です/デス/助動詞'])`

## Want to try more examples of the model?
go to http://nlpfun.com/ime
