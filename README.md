[TensorFlow 2] Xception: Deep Learning with Depthwise Separable Convolutions
=====
TensorFlow implementation of "Xception: Deep Learning with Depthwise Separable Convolutions"

## Related Repositories
<a href="https://github.com/YeongHyeon/Inception_Simplified-TF2">Inception_Simplified-TF2</a>  

## Concept
<div align="center">
  <img src="./figures/inception_xception.png" width="500">  
  <p>The Xception module comparing with the Inception module [1].</p>
</div>

<div align="center">
  <img src="./figures/inception.png" width="400">
  <img src="./figures/xception.png" width="400">  
  <p>Comparing the Inception and the Xception module via 3D view.</p>
</div>

## Performance

|Indicator|Value|
|:---|:---:|
|Accuracy|0.99480|
|Precision|0.99469|
|Recall|0.99486|
|F1-Score|0.99477|

```
Confusion Matrix
[[ 977    0    0    0    0    0    1    0    1    1]
 [   1 1127    0    0    3    0    1    3    0    0]
 [   1    0 1026    0    0    0    1    2    2    0]
 [   0    0    2 1001    0    5    0    0    2    0]
 [   0    0    0    0  979    0    0    0    1    2]
 [   0    0    0    3    0  888    1    0    0    0]
 [   2    0    0    0    0    1  955    0    0    0]
 [   0    1    1    0    0    0    0 1024    1    1]
 [   1    0    0    0    0    0    0    0  972    1]
 [   0    0    0    0    6    2    0    1    1  999]]
Class-0 | Precision: 0.99491, Recall: 0.99694, F1-Score: 0.99592
Class-1 | Precision: 0.99911, Recall: 0.99295, F1-Score: 0.99602
Class-2 | Precision: 0.99708, Recall: 0.99419, F1-Score: 0.99563
Class-3 | Precision: 0.99701, Recall: 0.99109, F1-Score: 0.99404
Class-4 | Precision: 0.99089, Recall: 0.99695, F1-Score: 0.99391
Class-5 | Precision: 0.99107, Recall: 0.99552, F1-Score: 0.99329
Class-6 | Precision: 0.99583, Recall: 0.99687, F1-Score: 0.99635
Class-7 | Precision: 0.99417, Recall: 0.99611, F1-Score: 0.99514
Class-8 | Precision: 0.99184, Recall: 0.99795, F1-Score: 0.99488
Class-9 | Precision: 0.99502, Recall: 0.99009, F1-Score: 0.99255

Total | Accuracy: 0.99480, Precision: 0.99469, Recall: 0.99486, F1-Score: 0.99477
```

## Requirements
* Python 3.7.6  
* Tensorflow 2.1.0  
* Numpy 1.18.1  
* Matplotlib 3.1.3  

## Reference
[1] François Chollet (2016). <a href="https://arxiv.org/abs/1610.02357">Xception: Deep Learning with Depthwise Separable Convolutions</a>. arXiv preprint arXiv:1610.02357.
