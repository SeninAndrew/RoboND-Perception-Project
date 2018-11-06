# Perception project report

This is submission of the "3D Perception" project for Udacity SE Robotics Nano degree. This includes exercised 1, 2, 3 and the final project. As I'm late with my submission I decided not to do the "challenge" part, only the basic tasks required to pass the project. 

## Exercise 1. Pipeline for filtering and RANSAC plane fitting implemented
See [RANSAC.py](https://github.com/SeninAndrew/RoboND-Perception-Project/blob/master/Exercise-1/RANSAC.py).

## Exercise 2. Pipeline including clustering for segmentation implemented
See [segmentation.py](https://github.com/SeninAndrew/RoboND-Perception-Project/blob/master/Exercise-2/sensor_stick/scripts/segmentation.py).

## Exercise 3. Features extracted and SVM trained. Object recognition implemented
See [object_recognition.py](https://github.com/SeninAndrew/RoboND-Perception-Project/blob/master/Exercise-3/sensor_stick/scripts/object_recognition.py) and [train_svm.py](https://github.com/SeninAndrew/RoboND-Perception-Project/blob/master/Exercise-3/sensor_stick/scripts/train_svm.py).

![Example 3 confusion matrix](https://github.com/SeninAndrew/RoboND-Perception-Project/blob/master/imgs/Example3_confusion.png)

## Project: Pick and Place Setup

I'm using the same training parameter as in the previous excercise. I replaced models at Exercise-3/sensor_stick/scripts/capture_features.py from:

```python
    models = [\
       'beer',
       'bowl',
       'create',
       'disk_part',
       'hammer',
       'plastic_cup',
       'soda_can']
```
         
to (see Exercise-3/sensor_stick/scripts/capture_features_p2.py): 

```python
   models = [
       'biscuits',
       'book',
       'eraser',
       'glue',
       'snacks',
       'soap',
       'soap2',
       'soda_can',
       'sticky_notes',
       ]
```

Then run capturing features and model training. 

Then I updated the code at https://github.com/SeninAndrew/RoboND-Perception-Project/blob/master/pr2_robot/scripts/object_recognition.py to segment object, calculate features and run classification. 

With that model I managed to achieve the following classification results on the 3 worlds:

World 1:
![World 1 image](https://github.com/SeninAndrew/RoboND-Perception-Project/blob/master/imgs/project_1.png)

The resulting [output file](https://github.com/SeninAndrew/RoboND-Perception-Project/blob/master/pr2_robot/scripts/output_1.yaml).

World 2:
![World 2 image](https://github.com/SeninAndrew/RoboND-Perception-Project/blob/master/imgs/project_2.png)

The resulting [output file](https://github.com/SeninAndrew/RoboND-Perception-Project/blob/master/pr2_robot/scripts/output_2.yaml).

World 3:
![World 3 image](https://github.com/SeninAndrew/RoboND-Perception-Project/blob/master/imgs/project_3.png)

The resulting [output file](https://github.com/SeninAndrew/RoboND-Perception-Project/blob/master/pr2_robot/scripts/output_3.yaml).
