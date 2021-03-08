# A series of detections based on opencv and tensorflow running on RaspiberryPi 4B

This is a series of detections based on opencv and tensorflow running on RaspiberryPi 4B.

When the program is running,it loads the dependencies this program needs such as some Cascade Classifier,
a typical object detection model(ssd_mobilenet_v1_coco_2018_01_28) and a typical text detection model(east model).
Then it will recognize your gesture(1 to 4), and choose a detection by it.

1->object detection

2->text detection

3->face detection(including eyes and smiles)

4->moving detection

A submodule of this program is /models from [Model Garden for TensorFlow](https://github.com/tensorflow/models) 
and adding a third-party trained model ssd_mobilenet_v1_coco_2018_01_28
[ssd_mobilenet_v1_coco_2018_01_28](download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz)

## Dependencies
| Dependency        | OS Version | RaspiberryPi Version|
|-------------------|------------|---------------------|
| python            | 3.7.3      | 3.7.3               |
| tensorflow        | 2.4.1      | 1.14.0              |
| opencv-python	    | 4.5.1.48   | 4.5.0               |
| imutils           | 0.5.4      | 0.5.4               |



## Tips
+ if the tensorflow model didn't work:
    add the "TensorFlow-ObjectDetection-models.pth" file to python/site-packages.

    such as "/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages"