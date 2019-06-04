# JLM
A fast LSTM Language Model for large vocabulary language like Japanese and Chinese.

### Faster and smaller without accuracy loss 

It focuses on **accelerating inference time** and **reducing model size** to fit requirement of real-time applications especially in client side. It is **85% smaller**, and are **50x faster** than standard LSTM solution with softmax. See the paper [JLM - Fast RNN Language Model with Large Vocabulary](http://anlp.jp/proceedings/annual_meeting/2018/pdf_dir/D3-4.pdf) for performance detail.

### No dependency to training framework
The training part is done with TensorFlow. Instead of depending on a big dynamic library of TF to run in client app, we dumped the trained weights out. The inference and decoding can be done by python with numpy or C++ with Eigen. There is no black box.

## Language model
We implemented the standard LSTM , [tie-embedding](https://arxiv.org/abs/1608.05859), [D-softmax](https://arxiv.org/abs/1512.04906), and [D-softmax*](https://arxiv.org/abs/1609.04309) in both training and numpy inference stages as a comparison. In practice, please consider just use D-softmax* for your best interest.

## Decoder
A standard Viterbi decoder with beam search is implemented. It batches prediction to save decoding time. We also implemented [Enabling Real-time Neural IME with Incremental Vocabulary Selection](https://www.aclweb.org/anthology/N19-2001) at NAACL 2019. It further more reduced the softmax cost during decoding by ~95%, reaching real-time in commodity CPU. 

# How to use
## Corpus preparation
We use [BCCWJ](http://pj.ninjal.ac.jp/corpus_center/bccwj/en/) Japanese corpus for example. 

Make sure you have a txt file with a white space segmented sentence per line.
> If not, use tools like [MeCab](http://taku910.github.io/mecab/) to get the job done first.

In JLM, the unit **word** is the in the format of **display/reading/POS**.  For example "品川/シナガワ/名詞-固有名詞-地名-一般". 

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
    "num_steps": 20,
    "max_epochs": 10,
    "early_stopping": 1,
    "dropout": 0.9,
    "lr": 0.001,
    "share_embedding": True,
    "gpu_id": 0,
    "tf_random_seed": 101,
    "D_softmax": False,
    "V_table": True,
    "embedding_seg": [(200, 0, 12000), (100, 12000, 30000), (50, 30000, None)]
}
```
 **V_table** is the D-softmax* here. The **embedding_seg** means how you want to separate your vocabulary. Make your own best result by tuning the hyper parameters. The results will be saved to folder "experiments" with a ID, **sacred** framework will take care of all the experiment indexing.
 > A small hint here, use up your GPU memory by setting a bigger **num_steps** and **batch_size** to best reduce training time.

## Verify LM is correct
Run the [test.py](https://github.com/jiali-ms/JLM/blob/master/train/test.py) in train folder, it will auto generate a random sentence with the trained model. If the sentence doesn't make sense to your language knowledge at all. One of the stage is not correctly setup. Here is an example of random generated sentence.
> 自宅 に テレビ 局 で 電話 し たら 、 で も 、 すべて の 通話 が 実現 。

## Dump  the TF trained weights
It is a common issue to bring trained model into clients like mobile device or Windows environment. We recommend you use ONNX for the purpose. But for this example, we want **full** control and want to cutoff dependency to TF. Run the [weights.py](https://github.com/jiali-ms/JLM/blob/master/train/weights.py) in the train folder to get a pickle of all the trainable parameters in the numpy format. You can use the following command to even dump the txt format for other usage.
> python weiths.py -e 1 -v True

## Evaluation
Run [eval.py](https://github.com/jiali-ms/JLM/blob/master/decoder/eval.py) to see the conversion accuracy of your trained model with offline model. 
> python eval.py

## Compression
We also implemented the k-means quantization mentioned in the ICLR 2016 paper [Deep Compression](https://arxiv.org/pdf/1510.00149v5.pdf). The code is at [comp.py](https://github.com/jiali-ms/JLM/blob/master/train/comp.py). Run the script with the experiment id. It will generate the code and codebook. By our experiments with 8 bits, there is almost no accuracy loss for conversion task. 

## Want to try more examples of the model?
go to http://nlpfun.com/ime
