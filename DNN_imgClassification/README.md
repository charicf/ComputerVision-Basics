# Deep Neural Network to classify images from Fashion MNIST - ComputerVision
The purpose is to explore the use of Deep Neural Networks (DNN’s) and their practises in Computer Vision. It will train and evaluate a DNN to classify images from a widely available and popular dataset, Fashion MNIST. The objective is to classify an unknown sample in one of the 10 classes. Its performance is evaluated on the popular metrics of Accuracy, Precision and Recall.

## Usage
### image_classification.py

```python
python image_classification.py <args>
'--lr',    type = float,metavar = 'lr',   default='0.001',help="Learning rate for the oprimizer."
'--m',     type = float,metavar = 'float',default= 0,     help="Momentum for the optimizer, if any."
'--bSize', type = int,  metavar = 'bSize',default=32,     help="Batch size of data loader, in terms of samples. a size of 32 means 32 images for an optimization step."
'--epochs',type = int,  metavar = 'e',    default=12   ,  help="Number of training epochs. One epoch is to perform an optimization step over every sample, once."
    

```

## Contributing
Pull requests are welcome.

## License
[MIT](https://choosealicense.com/licenses/mit/)