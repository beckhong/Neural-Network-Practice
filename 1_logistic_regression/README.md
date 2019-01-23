# Logistic regression with Deep Learning

## Sigmoid Function
![Sigmoid](http://latex.codecogs.com/gif.latex?f%28z%29%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-z%7D%7D)


## Calculate the Output of a Neuron
![Simple neuron](http://imagebank.osa.org/getImage.xqy?img=QC5sYXJnZSxvZS0xNC0xNC02NDU2LWcwMDE) 

(reference: https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-14-14-6456&id=90787)


## Bayes' Rule
![Bayes' Rule](http://latex.codecogs.com/gif.latex?P%28Y%7CX%29%20%3D%20%5Cfrac%7BP%28X%7CY%29P%28Y%29%7D%7BP%28X%29%7D)


##  Bayes Binary Classifier
### Target
We train a model on inputs X and targets Y i.e. ![distributed](http://latex.codecogs.com/gif.latex?P%28X%7CY%29).

Suppose that class Y in (0, 1).
```
![Bayes Classifier1](http://latex.codecogs.com/gif.latex?P%28Y%3D1%7CX%29%20%3D%20%5Cfrac%7BP%28X%7CY%3D1%29P%28Y%3D1%29%7D%7BP%28X%29%7D%20%3D%20%5Cfrac%7BP%28X%7CY%3D1%29P%28Y%3D1%29%7D%7BP%28X%7CY%3D0%29P%28Y%3D0%29&plus;P%28X%7CY%3D1%29P%28Y%3D1%29%7D)

![Bayes Classifier2](http://latex.codecogs.com/gif.latex?P%28Y%3D1%7CX%29%20%3D%20%5Cfrac%7B1%7D%7B1&plus;%5Cfrac%7BP%28X%7CY%3D0%29P%28Y%3D0%29%7D%7BP%28X%7CY%3D1%29P%28Y%3D1%29%7D%7D%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E%7B-%28W%5E%7BT%7DX&plus;b%29%7D%7D)

![Bayes Classifier3](http://latex.codecogs.com/gif.latex?-%28W%5E%7BT%7DX&plus;b%29%20%3D%20ln%28%5Cfrac%7BP%28X%7CY%3D0%29P%28Y%3D0%29%7D%7BP%28X%7CY%3D1%29P%28Y%3D1%29%7D%29)
```


## Cost Function in Logistic Regression
### Cross-Entropy Error
```
![Cross-Entropy](http://latex.codecogs.com/gif.latex?J_n%20%3D%20-%28t_%7Bn%7Dlog%28y_%7Bn%7D%29&plus;%281-t_%7Bn%7D%29log%281-y_%7Bn%7D%29%29)
```
t: target; y: output of logistic

### Multiple Training Error
```
![Multiple](http://latex.codecogs.com/gif.latex?J%20%3D%20-%5Csum_%7Bn%3D1%7D%5E%7BN%7D%20%28t_%7Bn%7Dlog%28y_%7Bn%7D%29&plus;%281-t_%7Bn%7D%29log%281-y_%7Bn%7D%29%29)
```
