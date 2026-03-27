# Report Notes

## 1

### 1.1

$\square$ **Prediction 1**: Before training, write whether you think this model will overfit, and briefly justify.

Notes
- (UF) feature quality: the character frequency counts drop all sequence information
- (OF) We have a small training dataset: 10.5k samples
- (UF) Relatively few parameters: 1901, limited capacity to memorize data
- (OF) 500 epochs, the model is optimized on each sample many times
- (OF) SGD is the only regularizing force, and the batches are fairly large (64 samples)

It will overfit. However, if we assume the character sequence contains useful predictive signal the loss will remain high.

### 1.2

$\square$ **Prediction 2**: Before the experiment, write which optimizer will converge fastest, and why.

Options:
- SGD
- SGD with momentum
- Adam

Adam will converge fastest, followed by SGD with momentum, then SGD. Training with SGD took a while to get going. Momentum will help that but may cause instability while converging.