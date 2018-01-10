## Model Training  

After the corpus and all the pickles are prepared, start from the "train" file, update the config and run. The results and console output will be in experiment folder with an experiment ID. If the result is good to release, run the "weights" script with that ID. Raw text weights and pickle version will be generated for decoder.  

### model.py  
The LSTM model definition that construct the tensorflow graph. It is a standard LSTM implmentation with a config to share the input and output embeddings.  

### train.py  
The main entrance for the training process. It uses sacred framework to record each experiment with different config.  

### test.py  
A sentence generator to check if the model is correctly trained.    
 
### weights.py   
Load the dumped tensorflow version weights and convert to raw text and pickle

### utils.py    
Utils for model, train and test
