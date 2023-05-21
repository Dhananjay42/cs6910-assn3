# CS6910 - Fundamentals of Deep Learning: Assignment-3
By Dhananjay Balakrishnan, ME19B012. Submitted as part of the course CS6910, taught by Professor Mitesh Khapra, Jan-May 2023 semester.

The models can be found here: https://drive.google.com/drive/folders/1F8eWN2EU41L3-NFPbWg1G6-izWT2MMpP?usp=sharing.

The detailed wandb report for this project can be found here: https://wandb.ai/clroymustang/cs6910-assignment-3/reports/CS6910-Assignment-3-Recurrent-Neural-Networks--Vmlldzo0MzI4NjY3?accessToken=qtmhvgphngdiexi346ilv6n7b5bvma44d94u6lf6o01nvojt8bgslxs91tjck4h2. 

## Instructions to use:
### Training the Neural Network.
1. To train the Neural Network, use the 'train.py' script. Here is the format of how you should run it:

```
python train.py <arguments>
```
The Compulsory Arguments are:
|Argument Name|Description|Default Value|
| ------------- | ------------- | -------- |
|-d, --dataset|Link to the dataset directory|''|
|-l, --language_prefix|To access files of the type prefix+'_train.csv', etc. |''|

The optional arguments are:
|Argument Name|Description|Default Value|
| ------------- | ------------- | -------- |
|-i, --n_iterations|Number of iterations.|75000|
|-a, --attention|Choose from 'y' or 'n'|'n'|
|-b, --beam_size|Beam Size (only applicable without attention)|1|
|-e, --embedding_size|Embedding Size|128|
|-h, --hidden_size|Hidden Layer Size|128|
|-nl, --n_layers|Number of Layers (not applicable for attention)|3|
|-c, --cell_type|Cell Type. Choose from ['LSTM', 'GRU', 'RNN']. |LSTM/GRU w and w/o attn respectively|
|-lr, --learning_rate|Learning Rate|0.01 w/o attn, 0.005 w attn|
|-q, --quiet|choose from 'y' or 'n'. whether you want the model to print progess|'y'|
|-d, --decay_rate|LR decay rate|0 w/o atten, log2(0.2/30000) w attn|
|-t, --teacher_forcing|Teacher Forcing Ratio.|0.5|
|-bi, --bidirectional|choose between 'y' or 'n'|'n' w/o attn, 'y' w attn|

2. On completion, the model will be stored in the same directory as 'model.pkl'. You can save the model to recreate the results later. 

## Assignment Specific Functions and WandB sweep
The WandB hyperparameter sweep, and all required comparisons for the various questions of the problem statement have been performed in the Jupyter Notebook titled 'assn3_nb.ipynb'.
