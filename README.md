# SimSiam Tensorflow

Minimal Implementation of a SimSiam (Simple Siamese Representation Learning) in tensorflow. checkout the  following paper for more details: [SimSiam: Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566). 


## Architecture

<div align="center">
  <img src="https://user-images.githubusercontent.com/91251307/215079798-efccb85b-a52a-4214-8792-5b13cb2af541.png" width="60%"/>
</div>

 - SimSiam architecture. Two augmented views of one
image are processed by the same encoder network f (a backbone
plus a projection MLP). Then a prediction MLP h is applied on one
side, and a stop-gradient operation is applied on the other side. The
model maximizes the similarity between both sides. It uses neither
negative pairs nor a momentum encoder.

## Pseudocode
```python
# f: backbone + projection mlp
# h: prediction mlp
for x in loader: # load a minibatch x with n samples
x1, x2 = aug(x), aug(x) # random augmentation
z1, z2 = f(x1), f(x2) # projections, n-by-d
p1, p2 = h(z1), h(z2) # predictions, n-by-d
L = D(p1, z2)/2 + D(p2, z1)/2 # loss
L.backward() # back-propagate
update(f, h) # SGD update
def D(p, z): # negative cosine similarity
z = z.detach() # stop gradient
p = normalize(p, dim=1) # l2-normalize
z = normalize(z, dim=1) # l2-normalize
return -(p*z).sum(dim=1).mean()
```

## Dataset
[STL10](https://www.kaggle.com/jessicali9530/stl10) dataset, is used to train the SimSiam Network(which has 100000 unlabeled data). And for the Linear Evaluation of SimSiam Model, Cifar10 dataset,

## Loss Function

The Loss function is the sum of the negative cosine similarity between the representations and the predictions of opposite branches:

$$Loss = \frac{1}{2} D(p_1, z_2) +  \frac{1}{2} D(p_2, z_1) \ \ with \ \ D(p, z) = - \frac{p \cdot z}{||p||_2 \cdot ||z||_2}$$


## Downstream Task

The downstream task of image classification on the [cifar 10](https://www.kaggle.com/c/cifar-10).

## Citations
```
@Article{chen2020simsiam,
  author  = {Xinlei Chen and Kaiming He},
  title   = {Exploring Simple Siamese Representation Learning},
  journal = {arXiv preprint arXiv:2011.10566},
  year    = {2020},
}
```
