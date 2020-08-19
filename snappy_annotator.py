"""
Annotation tool specifically designed for snappy annotation on object detection methods
"""

import anntoolkit
import imageio
import os
import numpy as np
import random
import cv2
from voc_save_load import save_to_voc, load_from_voc


def load_configs():
    lib_path = ''
    db = 'Unknown'
    default_lbl = '1'
    if os.path.exists(os.path.join('configurations', 'configs.txt')):
        with open(os.path.join('configurations', 'configs.txt'), 'r') as c:
            for line in c.readlines():
                line = line.strip()
                if line.startswith('LIBRARY_PATH:'):
                    lib_path = line[13:]
                if line.startswith('DATABASE:'):
                    db = line[9:]
                if line.startswith('DEF_LABEL:'):
                    default_lbl = line[10:]
    return lib_path, db, default_lbl


def load_classes():
    class_keys = []
    if os.path.exists(os.path.join('configurations', 'classes.txt')):
        with open(os.path.join('configurations', 'classes.txt'), 'r') as c:
            for line in c.readlines():
                class_keys.append(line.strip())
    return class_keys


def reset_box(bbox):
    xmin = min(bbox[0][0], bbox[1][0])
    xmax = max(bbox[0][0], bbox[1][0])
    ymin = min(bbox[0][1], bbox[1][1])
    ymax = max(bbox[0][1], bbox[1][1])
    return [(xmin, ymin), (xmax, ymax)]


class App(anntoolkit.App):
    def __init__(self):
        super(App, self).__init__(title='Snappy Annotator')

        self.path, self.database, self.def_label = load_configs()
        self.paths = []
        for dirName, subdirList, fileList in os.walk(self.path):
            self.paths += [os.path.relpath(os.path.join(dirName, x), self.path) for x in fileList if x.endswith('.jpg')
                           or x.endswith('.jpeg') or x.endswith('.png')]
        self.paths.sort()
        self.iter = -1
        self.classes = load_classes()
        self.labels = {}
        self.labels_on = True
        self.annotation = {}
        self.moving = None
        self.load_next()

        print("Number of annotated images: %d" % len(self.annotation.items()))
        print(self.annotation)

    # Creating due to fact that this appears in multiple locations: changing
    # dataset layout may require referencing a file's full path differently
    def get_annotation_path(self):
        k = self.paths[self.iter]
        return os.path.join(self.path, str(k[:k.find('.')]) + '_annotations.xml')

    # Loads in the annotations/labels for the current image - currently done every time next image is loaded in
    def load_current_annotations(self):
        k = self.paths[self.iter]
        _, _, _, anns, lbls = load_from_voc(self.path, k)
        self.annotation[k] = anns
        self.labels[k] = lbls

    # If the current sample contains an empty annotation, remove
    # it from the annotation list and delete the annotation file
    def remove_zero_annotations(self):
        k = self.paths[self.iter]
        if k in self.annotation and self.annotation[k] == []:
            self.annotation.pop(k)
            if os.path.exists(self.get_annotation_path()):
                os.remove(self.get_annotation_path())

    def load_next(self):
        self.remove_zero_annotations()
        self.iter += 1
        self.iter = self.iter % len(self.paths)
        im = imageio.imread(os.path.join(self.path, self.paths[self.iter]))
        self.set_image(im)
        self.load_current_annotations()

    def load_prev(self):
        self.remove_zero_annotations()
        self.iter -= 1
        self.iter = (self.iter + len(self.paths)) % len(self.paths)
        im = imageio.imread(os.path.join(self.path, self.paths[self.iter]))
        self.set_image(im)
        self.load_current_annotations()

    def load_next_not_annotated(self):
        self.remove_zero_annotations()
        while True:
            self.iter += 1
            self.iter = self.iter % len(self.paths)
            k = self.paths[self.iter]
            if k not in self.annotation:
                break
            if self.iter == 0:
                break
        try:
            im = imageio.imread(os.path.join(self.path, self.paths[self.iter]))
            self.set_image(im)
            self.load_current_annotations()
        except ValueError:
            self.load_next_not_annotated()

    def load_prev_not_annotated(self):
        self.remove_zero_annotations()
        while True:
            self.iter -= 1
            self.iter = (self.iter + len(self.paths)) % len(self.paths)
            k = self.paths[self.iter]
            if k not in self.annotation:
                break
            if self.iter == 0:
                break
        try:
            im = imageio.imread(os.path.join(self.path, self.paths[self.iter]))
            self.set_image(im)
            self.load_current_annotations()
        except ValueError:
            self.load_prev_not_annotated()

    def get_image_dims(self):
        k = self.paths[self.iter]
        img = cv2.imread(os.path.join(self.path, k))
        return img.shape

    # Resets each bounding box within the current image to prepare for Pascal VOC format - dimensions are rounded
    # to integers and smaller values of x and y are placed in first point, larger values in second point
    def reset_annotation_boxes(self):
        k = self.paths[self.iter]
        annotation_k = self.annotation[k]
        reset_annotation = []
        for i in range(0, len(annotation_k) - 1, 2):
            xmin = int(round(min(annotation_k[i][0], annotation_k[i+1][0])))
            xmax = int(round(max(annotation_k[i][0], annotation_k[i+1][0])))
            ymin = int(round(min(annotation_k[i][1], annotation_k[i+1][1])))
            ymax = int(round(max(annotation_k[i][1], annotation_k[i+1][1])))
            reset_annotation.append((xmin, ymin))
            reset_annotation.append((xmax, ymax))
        return reset_annotation

    # NOTE: Update to change the selected label
    def change_previous_label(self, key):
        num = int(key)
        if num > 0:  # 1 will be first item (index = 0), 2 will be second (index = 1), ...
            num -= 1
        else:  # 0 will be last, i.e. 10th, item (index = 9)
            num = 9
        if num < len(self.classes):
            k = self.paths[self.iter]
            if k in self.labels:
                self.labels[k][-1] = self.classes[num]
                save_to_voc(k, self.path, os.getcwd(), self.database, self.get_image_dims(),
                            self.reset_annotation_boxes(), self.labels[k])

    # Called once per frame. This is where things (including labels) are drawn on the image.
    def on_update(self):
        k = self.paths[self.iter]
        self.text(k, 10, 20)
        if k in self.annotation:
            self.text("Points count: %d" % len(self.annotation[k]), 10, 50)
            for i, p in enumerate(self.annotation[k]):
                if i == self.moving:
                    self.point(*p, (0, 255, 0, 250))
                else:
                    self.point(*p, (255, 0, 0, 250))

            n = 2
            boxes = [self.annotation[k][i:i + n] for i in range(0, len(self.annotation[k]), n)]
            for i, box in enumerate(boxes):
                if len(box) == 2 and k in self.labels:
                    # Colors implemented for first 4 labels. More can be implemented if desired; can also change colors
                    if self.labels[k][i] == self.classes[0]:
                        self.box(box, (0, 255, 0, 255), (0, 255, 0, 128))
                    elif self.labels[k][i] == self.classes[1]:
                        self.box(box, (255, 0, 0, 255), (255, 0, 0, 128))
                    elif self.labels[k][i] == self.classes[2]:
                        self.box(box, (249, 21, 218, 255), (249, 21, 218, 128))
                    elif self.labels[k][i] == self.classes[3]:
                        self.box(box, (255, 128, 0, 255), (255, 128, 0, 128))
                    else:
                        self.box(box, (0, 255, 0, 250), (100, 255, 100, 50))
                    if self.labels_on:
                        box = reset_box(box)
                        self.text_loc(self.labels[k][i], *box[0], (0, 10, 0, 250), (150, 255, 150, 150))

    def on_mouse_button(self, down, x, y, lx, ly):
        k = self.paths[self.iter]

        if down:
            if k in self.annotation:
                points = np.asarray(self.annotation[k])
                if len(points) > 0:
                    point = np.asarray([[lx, ly]])
                    d = points - point
                    d = np.linalg.norm(d, axis=1)
                    i = np.argmin(d)
                    if d[i] < 6:
                        self.moving = i
                        print(i, d[i])

        if not down:
            if k not in self.annotation:
                self.annotation[k] = []
            if self.moving is not None:
                self.annotation[k][self.moving] = (lx, ly)
                self.moving = None
            else:
                self.annotation[k].append((lx, ly))
                if len(self.annotation[k]) % 2 == 0:
                    if k not in self.labels:
                        self.labels[k] = [self.def_label]
                    else:
                        self.labels[k].append(self.def_label)
                    save_to_voc(k, self.path, os.getcwd(), self.database, self.get_image_dims(),
                                self.reset_annotation_boxes(), self.labels[k])

    def on_mouse_position(self, x, y, lx, ly):
        if self.moving is not None:
            k = self.paths[self.iter]
            self.annotation[k][self.moving] = (lx, ly)

    def on_keyboard(self, key, down, mods):

        if down:
            if key == anntoolkit.KeyLeft or key == 'A':
                self.load_prev()
            elif key == anntoolkit.KeyRight or key == 'D':
                self.load_next()
            elif key == anntoolkit.KeyUp or key == 'W':
                self.load_next_not_annotated()
            elif key == anntoolkit.KeyDown or key == 'S':
                self.load_prev_not_annotated()
            elif key == anntoolkit.KeyDelete:
                k = self.paths[self.iter]
                if k in self.annotation:
                    del self.annotation[k]
                if k in self.labels:
                    del self.labels[k]
                if os.path.exists(self.get_annotation_path()):
                    os.remove(self.get_annotation_path())
            elif key == anntoolkit.KeyBackspace:
                k = self.paths[self.iter]
                if k in self.annotation and len(self.annotation[k]) > 0:
                    self.annotation[k] = self.annotation[k][:-1]
                    self.remove_zero_annotations()
                if k in self.annotation and len(self.annotation[k]) % 2 == 1:
                    self.labels[k].pop()
                    save_to_voc(k, self.path, os.getcwd(), self.database, self.get_image_dims(),
                                self.reset_annotation_boxes(), self.labels[k])
            elif key == 'R':
                self.iter = random.randrange(len(self.paths))
                self.load_next()
            elif key == 'T':  # 'T' to toggle the labels on or off
                self.labels_on = not self.labels_on
            elif str(key).isnumeric():
                self.change_previous_label(key)


if __name__ == '__main__':
    snappy_annotator = App()
    snappy_annotator.run()
