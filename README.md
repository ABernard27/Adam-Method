# Adam : adaptative moment estimation

In this repository, we'll put our research and code on the ADAM method 
thanks to the article : [ADAM: A METHOD FOR STOCHASTIC 
OPTIMIZATION](https://browse.arxiv.org/pdf/1412.6980.pdf)

### Author
CAPEL Alexandre & BERNARD Anne, students at University of Montpellier

### Requirements 

To compute the file [script.py](script.py) you have to install some package. You can install them with this command in your terminal :
```bash
pip install -r requirements.txt
```


## Why?

*Adam* is a method for efficient stochastic optimization. 

During gradient descent, it is sometimes challenging to find the right learning rate. In fact, most of the time, this rate is fixed, and if it is too small, it can be difficult to find the minimum because the method won't progress enough, and it might stop before reaching it. If it is too large, we will progress quickly, but it is possible to go around the point without reaching it because the step is too large. Therefore, it is necessary to find the perfect step size to reach the minimum. 

Another point : when we have a big dataset we can't calculate the gradient with the entire dataset, that's too expensive. So we have to use just subsamples to reduce the computing time. We use many batch and many epoch to browse the entire dataset. 

## Example

We used the function `torch.optim.adam` in the `pytorch` library to compute the loss of two examples. The results are in the directory [graph](graph). You can find the script used in [script.py](script.py)

### Multinomial regression

We used the unavoidable dataset `MNIST`. We used the negativ log-likelihood of the multinomial as the objectiv function. The batch size is 128.

### Neural Network 

The neural network used is composed of 2 hidden layers and the output neural is linear. The objective function is a cross entropy. The batch size is also 128.

