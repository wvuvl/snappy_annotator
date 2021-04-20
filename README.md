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
- Set annotation class: '1'-'0' on keyboard (limit of 10/can be edited using 'configurations/classes.txt')
- Select annotation: left click while hovering annotation
- Select previous annotation: 'q'
- Select next annotation: 'e'
- Toggle labels (on screen) on/off: 't'
- Remove selected annotation: backspace
- Remove all annotations for image: delete
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

1. In 'classes.txt', the default classes can be set, where the first line references key 1, the second references key 2, and so on, with the 10th referencing key 
2. In 'configs.txt', the current default configurations are the path of the directory containing the files, the database to reference in the .xml files produced, and the default label when the program starts up. An additional line, "DB_CHANGED:<boolean>" (simply replace <boolean> with "True" or "False"), can be inserted to control whether or not file sorting needs to occur at the beginning of the function. The default, False, is to only sort if the corresponding "sorted_filenames_by_species.pkl" file exists in the root directory.
    - For the current setup, species names are used to sort images. For faster startup, a function is used to save this sorted order. This can easily be removed in the code by deleting the line above the commented line in \_\_init\_\_ in snappy_annotator.py, and uncommenting the commented line.

## Current Features

- Saving and loading in Pascal VOC format
- Configurations can be saved to file to remove changing settings at startup each time
- Keybindings for quick labeling using number keys - makes for less time per annotation
    - Note: code is set for using keys 1-0; current implementation uses keys 1-5
- Lists important information - current file name, keybindings, and number of annotations - on screen
- Annotations can be easily resized
- Annotations can be easily moved
- Annotations can be cycled through using keyboard presses, or simply selected using left click, in order to change previous labels quickly
- Annotations restricted to being within image dimensions
- Bounding box colors can be set for each label class (in code)
- Class labels for each annotation can be toggled on/off for viewing

