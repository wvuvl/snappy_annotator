# snappy_annotator
Annotation tool specifically built for training on Pascal VOC data. Built upon the framework of [AnnToolKit](https://github.com/podgorskiy/anntoolkit). 

## Quickstart: Controls

##### NOTE: Annotations are saved each time a label is changed or  whenever a box is created, deleted, moved, or resized. New annotations are saved with the same name as the image without the extension, with '\_annotations.xml' added to the end. Additionally, all changes on the current image can be undone by typing 'p', so long as the current image isn't changed to a different one.

- Set annotation (bounding box) point: left click
- Pan: right click
- Zoom: mouse wheel
- Change images: left key/right key (or 'a'/'d')
- Go to next/previous _annotated_ image: '.'/','
- Go to next/previous _un-annotated_ image: up key/down key (or 'w'/'s')
- Create annotation: left click to start creating bounding box, drag mouse to opposite desired corner, then left click again
- Set annotation class: '1'-'0' on keyboard (limit of 10/can be edited using 'configurations/classes.txt')
- Select annotation: left click while hovering annotation
- Select previous annotation: 'q'
- Select next annotation: 'e'
- Remove selected annotation: backspace/spacebar
- Remove all annotations for image: delete
- Toggle labels (on screen) on/off: 't'
- Rotate annotations (useful for when transitioning from old tool) 
    - 'u' for width-wise (centered about image's width), CW rotation
    - 'i' for height-wise, CW rotation
- Undo all changes to current image annotations: 'p' (see note above)

## Quickstart: Setup

To download, use the following lines of code:

    git clone https://github.com/wvuvl/snappy_annotator
    cd snappy_annotator
    pip install -r requirements.txt

For quicker startup, default settings are saved within two config files in the 'configurations' folder: 'classes.txt and configs.txt'.

1. In 'classes.txt', the default classes can be set, where the first line references key 1, the second references key 2, and so on, with the 10th referencing key 0.
2. In 'configs.txt', multiple keys can be used to configure various settings. These are the following:
 - 'LIBRARY_PATH:' - defines path to the database directory containing the images and annotations
 - 'DATABASE:' - the name of the database to be reflected in the metadata
 - 'SORT_BY_SPECIES:' - whether or not to sort the database by species
 - 'DB_CHANGED:' - Whether or not the database has been changed since last use. This value can be edited, or removing "sorted_filenames_by_species.pkl" from the root directory will do the same thing
 - 'OBSERVATION_RANK:' - The rank of the observation, to be reflected in the metadata. This is useful if doing multiple passes during annotation with the object detection-assisted annotator or somehow determining that certain observations contain a lower fidelity

The following entries are specific to the object detection-assisted annotation tool, snappy_OD_suggestions.py. As such, the values contained for them will not affect the standard snappy_annotator.py functionality.
 - 'PREDICTIONS_PATH:' - Location of 'coco_instances_results.json' file, containing json predictions which can be used by snappy_OD_suggestions.py
 - 'PREDICTION_THRESH:' - The threshold that bounding box prediction scores must be above in order to be considered
 - 'IOU_THRESH:' - The intersection-over-union threshold that bounding box proposals must be below in relation to each current annotation in order to be considered

## Current Features

- Saving and loading in Pascal VOC format
- Configurations can be saved to file to remove changing settings at startup each time
- Keybindings for quick labeling using number keys - makes for less time per annotation
    - Note: code is set for using keys 1-0; current implementation uses keys 1-5
- Lists important information - current file name/number, keybindings, and number of annotations - on screen
- Annotations can be easily resized
- Annotations can be easily moved
- Annotations can be cycled through using keyboard presses, or simply selected using left click, in order to change previous labels quickly
- Annotations restricted to being within image dimensions
- Bounding box colors can be set for each label class (in code)
- Class labels for each annotation can be toggled on/off for viewing

