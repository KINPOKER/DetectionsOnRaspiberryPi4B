# A series of detections based on opencv and tensorflow running on RaspiberryPi 4B

This is a series of detections based on opencv and tensorflow running on RaspiberryPi 4B.

When the program is running,it loads the dependencies this program needs such as some Cascade Classifier,
a typical object detection model(ssd_mobilenet_v1_coco_2018_01_28) and a typical text detection model(east model).
Then it will recognize your gesture(1 to 4), and choose a detection by it.

1->object detection

2->text detection

3->face detection(including eyes and smiles)

4->moving detection


## Dependencies
| Dependency        | Version   |
|-------------------|-----------|
| python            | 3.7.3     |
| tensorflow        | 2.4.1     |
| opencv-python	    | 4.5.1.48  |
| imutils           | 0.5.4     |


