"""
Annotation tool specifically designed for snappy annotation on object detection methods
"""

import anntoolkit
import imageio
import os
import numpy as np
import random
import cv2
from voc_save_load import save_to_voc_xml, load_from_voc_xml


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

        self.POINT_RADIUS = 6
        self.path, self.database, self.def_label = load_configs()
        self.paths = []
        for dirName, subdirList, fileList in os.walk(self.path):
            self.paths += [os.path.relpath(os.path.join(dirName, x), self.path) for x in fileList if x.endswith('.jpg')
                           or x.endswith('.jpeg') or x.endswith('.png')]
        self.paths.sort()
        self.iter = -1
        self.im_height = 0
        self.im_width = 0
        self.classes = load_classes()
        self.labels = {}
        self.labels_on = True
        self.annotations = {}
        self.new_box = None
        self.hovered_point = None
        self.moving_point = None
        self.hovered_box = -1
        self.moving_box = None
        self.selected_box_width = None
        self.selected_box_height = None
        self.highlighted = False
        self.selected_annot = -1
        self.load_next()

        # Cheating by doing this
        self.lx = 0
        self.ly = 0

    def get_image_dims(self):
        k = self.paths[self.iter]
        img = cv2.imread(os.path.join(self.path, k))
        return img.shape

    # If the current sample contains an empty annotation, remove
    # it from the annotation list and delete the annotation file
    def remove_zero_annotations(self):
        k = self.paths[self.iter]
        if k in self.annotations and self.annotations[k] == []:
            self.annotations.pop(k)
            if os.path.exists(self.get_annotation_path()):
                os.remove(self.get_annotation_path())

    # Loads in the annotations/labels for the current image, including height and width
    def load_current_im_info(self):
        k = self.paths[self.iter]
        _, _, anns, lbls = load_from_voc_xml(self.path, k)
        self.annotations[k] = anns
        self.labels[k] = lbls
        self.reset_highlight()
        self.im_height = self.get_image_dims()[0]
        self.im_width = self.get_image_dims()[1]

    def load_next(self):
        self.remove_zero_annotations()
        self.iter += 1
        self.iter = self.iter % len(self.paths)
        im = imageio.imread(os.path.join(self.path, self.paths[self.iter]))
        self.set_image(im)
        self.load_current_im_info()

    def load_prev(self):
        self.remove_zero_annotations()
        self.iter -= 1
        self.iter = (self.iter + len(self.paths)) % len(self.paths)
        im = imageio.imread(os.path.join(self.path, self.paths[self.iter]))
        self.set_image(im)
        self.load_current_im_info()

    def load_next_not_annotated(self):
        self.remove_zero_annotations()
        while True:
            self.iter += 1
            self.iter = self.iter % len(self.paths)
            self.load_current_im_info()
            k = self.paths[self.iter]
            if self.annotations[k] == [] or self.iter == 0:
                break
        try:
            im = imageio.imread(os.path.join(self.path, self.paths[self.iter]))
            self.set_image(im)
        except ValueError:
            self.load_next_not_annotated()

    def load_next_annotated(self):
        self.remove_zero_annotations()
        while True:
            self.iter += 1
            self.iter = self.iter % len(self.paths)
            self.load_current_im_info()
            k = self.paths[self.iter]
            if not self.annotations[k] == [] or self.iter == 0:
                break
        try:
            im = imageio.imread(os.path.join(self.path, self.paths[self.iter]))
            self.set_image(im)
        except ValueError:
            self.load_next_not_annotated()

    def load_prev_not_annotated(self):
        self.remove_zero_annotations()
        while True:
            self.iter -= 1
            self.iter = (self.iter + len(self.paths)) % len(self.paths)
            self.load_current_im_info()
            k = self.paths[self.iter]
            if self.annotations[k] == [] or self.iter == 0:
                break
        try:
            im = imageio.imread(os.path.join(self.path, self.paths[self.iter]))
            self.set_image(im)
            self.load_current_im_info()
        except ValueError:
            self.load_prev_not_annotated()

    def load_prev_annotated(self):
        self.remove_zero_annotations()
        while True:
            self.iter -= 1
            self.iter = (self.iter + len(self.paths)) % len(self.paths)
            self.load_current_im_info()
            k = self.paths[self.iter]
            if not self.annotations[k] == [] or self.iter == 0:
                break
        try:
            im = imageio.imread(os.path.join(self.path, self.paths[self.iter]))
            self.set_image(im)
            self.load_current_im_info()
        except ValueError:
            self.load_prev_not_annotated()

    def change_selected_label(self, key):
        num = int(key)
        if num > 0:  # 1 will be first item (index = 0), 2 will be second (index = 1), ...
            num -= 1
        else:  # 0 will be last, i.e. 10th, item (index = 9)
            num = 9
        if num < len(self.classes):
            k = self.paths[self.iter]
            self.def_label = self.classes[num]
            if k in self.labels and len(self.labels[k]) > 0:
                self.labels[k][self.selected_annot] = self.classes[num]
                save_to_voc_xml(k, self.path, os.getcwd(), self.database, self.get_image_dims(),
                                self.reset_annotation_boxes(), self.labels[k])

    # Created due to fact that this appears in multiple locations: changing
    # dataset layout may require referencing a file's full path differently
    def get_annotation_path(self):
        k = self.paths[self.iter]
        return os.path.join(self.path, str(k[:k.find('.')]) + '_annotations.xml')

    # NOTE: This is specifically used for PlantCLEF 2015 dataset format
    def get_PC15_metadata_category(self):
        k = self.paths[self.iter]
        xml = os.path.join(self.path, str(k[:k.find('.')]) + '.xml')
        if os.path.exists(xml):
            with open(xml, 'r') as x:
                for line in x.readlines():
                    if line.strip().startswith('<Content>'):
                        return line.strip()[9:-10]
            return '**no image label found**'
        return '**no metadata xml file found**'

    # Returns a list of the opposite corners of the original annotations, which is used to
    # create the second pair of points for each bounding box
    def get_ann_opposite_corners(self):
        k = self.paths[self.iter]
        opposite_corners = []
        for i in range(0, len(self.annotations[k]) - 1, 2):
            opposite_corners.append((self.annotations[k][i][0], self.annotations[k][i + 1][1]))
            opposite_corners.append((self.annotations[k][i + 1][0], self.annotations[k][i][1]))
        return opposite_corners

    # Resets variables when highlight is no longer visible: sets the selected annotation as the last one annotated
    def reset_highlight(self):
        k = self.paths[self.iter]
        self.highlighted = False
        if k in self.annotations and len(self.annotations[k]) > 0:
            self.selected_annot = int(len(self.annotations[k]) / 2) - 1
        else:
            self.selected_annot = 0

    # Resets each bounding box within the current image to prepare for Pascal VOC format - dimensions are rounded
    # to integers and smaller values of x and y are placed in first point, larger values in second point
    def reset_annotation_boxes(self):
        k = self.paths[self.iter]
        if k in self.annotations:
            annotation_k = self.annotations[k]
            reset_annotation = []
            for i in range(0, len(annotation_k) - 1, 2):
                xmin = int(round(min(annotation_k[i][0], annotation_k[i + 1][0])))
                xmax = int(round(max(annotation_k[i][0], annotation_k[i + 1][0])))
                ymin = int(round(min(annotation_k[i][1], annotation_k[i + 1][1])))
                ymax = int(round(max(annotation_k[i][1], annotation_k[i + 1][1])))
                reset_annotation.append((xmin, ymin))
                reset_annotation.append((xmax, ymax))
            if len(annotation_k) % 2 == 1:
                reset_annotation.append((int(round(annotation_k[-1][0])), int(round(annotation_k[-1][1]))))
            self.annotations[k] = reset_annotation
            return reset_annotation

    # Called once per frame. This is where things (including labels) are drawn on the image.
    def on_update(self):
        k = self.paths[self.iter]
        self.text(k, 10, 30)
        self.text("Metadata category: %s" % self.get_PC15_metadata_category(), 10, 60)
        self.text("Key bindings:", self.width-10, 30, alignment=anntoolkit.Alignment.Right)
        for i, c in enumerate(self.classes):
            self.text("{} - {}".format(i + 1, c), self.width-10, 70 + i * 30, alignment=anntoolkit.Alignment.Right)
        self.text("Current label: {}".format(self.def_label), 10, 90)
        if k in self.annotations:
            self.text("Points count: %d" % len(self.annotations[k]), 10, 120)
            for i, p in enumerate(self.annotations[k]):
                if i == self.hovered_point:
                    self.point(*p, (127, 127, 255, 159), radius=self.POINT_RADIUS*self.scale)
                self.point(*p, (255, 0, 0, 250))
            for i, p in enumerate(self.get_ann_opposite_corners()):
                self.point(*p, (255, 0, 0, 250))

            n = 2
            boxes = [self.annotations[k][i:i + n] for i in range(0, len(self.annotations[k]), n)]
            for i, box in enumerate(boxes):
                if len(box) == 2 and k in self.labels:
                    if self.hovered_box == i:
                        self.box(box, (255, 255, 255, 255), (255, 255, 255, 50))
                    if self.selected_annot == i and self.highlighted:  # When we are on selected box
                        self.box(box, (255, 255, 255, 255), (255, 255, 255, 128))
                    # Colors implemented for first 5 labels. More can be implemented if desired; can also change colors
                    elif self.labels[k][i] == self.classes[0]:
                        self.box(box, (0, 255, 0, 255), (0, 255, 0, 120))
                    elif self.labels[k][i] == self.classes[1]:
                        self.box(box, (255, 0, 0, 255), (255, 0, 0, 120))
                    elif self.labels[k][i] == self.classes[2]:
                        self.box(box, (249, 21, 218, 255), (249, 21, 218, 120))
                    elif self.labels[k][i] == self.classes[3]:
                        self.box(box, (255, 127, 0, 255), (255, 127, 0, 120))
                    elif self.labels[k][i] == self.classes[4]:
                        self.box(box, (127, 127, 127, 255), (127, 127, 127, 120))
                    else:
                        self.box(box, (0, 255, 0, 250), (100, 255, 100, 120))
                    if self.labels_on:
                        box = reset_box(box)
                        self.text_loc(self.labels[k][i], *box[0], (0, 10, 0, 250), (150, 255, 150, 150))
            if self.new_box:
                self.box(*self.new_box)

    def on_mouse_button(self, down, x, y, lx, ly):
        k = self.paths[self.iter]
        # Upon click
        if down:
            if k in self.annotations and not self.new_box:
                if self.hovered_point is not None:
                    self.moving_point = self.hovered_point
                elif self.hovered_box >= 0:
                    lower_diff = np.subtract((lx, ly), self.annotations[k][self.hovered_box * 2])
                    upper_diff = np.subtract((lx, ly), self.annotations[k][self.hovered_box * 2 + 1])
                    self.moving_box = [lower_diff, upper_diff]
                    self.selected_box_width = self.annotations[k][self.hovered_box * 2 + 1][0] - \
                                              self.annotations[k][self.hovered_box * 2][0]
                    self.selected_box_height = self.annotations[k][self.hovered_box * 2 + 1][1] - \
                                               self.annotations[k][self.hovered_box * 2][1]
                    self.selected_annot = self.hovered_box

        # Upon release
        if not down:
            if k not in self.annotations:
                self.annotations[k] = []
            if self.moving_box is not None:
                self.hovered_box = -1
                self.moving_box = None
                save_to_voc_xml(k, self.path, os.getcwd(), self.database, self.get_image_dims(),
                                self.reset_annotation_boxes(), self.labels[k])
                self.selected_box_height, self.selected_box_width = None, None
            elif self.moving_point is not None:
                self.annotations[k][self.moving_point] = (min(max(0, lx), self.im_width),
                                                          min(max(0, ly), self.im_height))
                self.moving_point = None
                save_to_voc_xml(k, self.path, os.getcwd(), self.database, self.get_image_dims(),
                                self.reset_annotation_boxes(), self.labels[k])
                self.hovered_point = None
            else:
                self.annotations[k].append((min(max(0, lx), self.im_width), min(max(0, ly), self.im_height)))
                if len(self.annotations[k]) % 2 == 0:
                    self.reset_highlight()
                    self.new_box = None
                    if k not in self.labels:
                        self.labels[k] = [self.def_label]
                    else:
                        self.labels[k].append(self.def_label)
                    save_to_voc_xml(k, self.path, os.getcwd(), self.database, self.get_image_dims(),
                                    self.reset_annotation_boxes(), self.labels[k])

    # Whenever the mouse changes position
    def on_mouse_position(self, x, y, lx, ly):
        self.lx = lx
        self.ly = ly
        k = self.paths[self.iter]
        if k in self.annotations:
            # Dragging point
            if self.moving_point is not None:
                self.annotations[k][self.moving_point] = (
                min(max(0, lx), self.im_width), min(max(0, ly), self.im_height))
            # Highlight hovered box: smallest box hovered will be highlighted
            elif self.moving_box is not None:
                # Limits movement of box to the inner bounds of the image
                lower_x = min(max(0, lx - self.moving_box[0][0]), self.im_width - self.selected_box_width)
                lower_y = min(max(0, ly - self.moving_box[0][1]), self.im_height - self.selected_box_height)
                upper_x = min(max(self.selected_box_width, lx - self.moving_box[1][0]), self.im_width)
                upper_y = min(max(self.selected_box_height, ly - self.moving_box[1][1]), self.im_height)
                self.annotations[k][self.hovered_box * 2] = (lower_x, lower_y)
                self.annotations[k][self.hovered_box * 2 + 1] = (upper_x, upper_y)
            elif not self.new_box:
                # Hovering box
                hovered_box_boxes = {}
                boxes = [self.annotations[k][i:i + 2] for i in range(0, len(self.annotations[k]) - 1, 2)]
                for i, b in enumerate(boxes):
                    b_xmin = min(b[0][0], b[1][0])
                    b_ymin = min(b[0][1], b[1][1])
                    b_xmax = max(b[0][0], b[1][0])
                    b_ymax = max(b[0][1], b[1][1])
                    if b_xmin <= lx <= b_xmax and b_ymin <= ly <= b_ymax:
                        x = b_xmax - b_xmin
                        y = b_ymax = b_ymin
                        hovered_box_boxes[i] = x * y
                if hovered_box_boxes:  # Finds index of the smallest hovered box
                    self.hovered_box = list(hovered_box_boxes.keys())[
                        list(hovered_box_boxes.values()).index(min(list(hovered_box_boxes.values())))]
                else:
                    self.hovered_box = -1
                # Hovering point
                points = np.asarray(self.annotations[k])
                opposite_points = np.asarray(self.get_ann_opposite_corners())
                if len(opposite_points) > 0:
                    point = np.asarray([[lx, ly]])
                    d_p = points - point
                    d_p = np.linalg.norm(d_p, axis=1)
                    ind_p = np.argmin(d_p)
                    d_op = opposite_points - point
                    d_op = np.linalg.norm(d_op, axis=1)
                    ind_op = np.argmin(d_op)
                    if d_p[ind_p] < d_op[ind_op] and d_p[ind_p] < self.POINT_RADIUS:
                        self.hovered_point = ind_p
                    elif d_op[ind_op] < self.POINT_RADIUS:
                        self.hovered_point = ind_op
                        ind_op = int(ind_op / 2) * 2  # Round down to even number
                        self.annotations[k][ind_op] = opposite_points[ind_op]
                        self.annotations[k][ind_op + 1] = opposite_points[ind_op + 1]
                    else:
                        self.hovered_point = None
            if len(self.annotations[k]) % 2 == 1:
                self.new_box = [
                    [self.annotations[k][-1], (min(max(0, lx), self.im_width), min(max(0, ly), self.im_height))],
                    (0, 0, 255, 192), (0, 0, 255, 128)]
            else:
                self.new_box = None

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
            elif key == ',':
                self.load_prev_annotated()
            elif key == '.':
                self.load_next_annotated()
            elif key == anntoolkit.KeyDelete:
                k = self.paths[self.iter]
                if k in self.annotations:
                    del self.annotations[k]
                if k in self.labels:
                    del self.labels[k]
                if os.path.exists(self.get_annotation_path()):
                    os.remove(self.get_annotation_path())
                self.reset_highlight()
            elif key == anntoolkit.KeyBackspace:
                k = self.paths[self.iter]
                if self.highlighted and len(self.annotations[k]) > 1:
                    self.annotations[k].pop(self.selected_annot * 2)
                    self.annotations[k].pop(self.selected_annot * 2)
                    self.selected_annot -= 1
                    save_to_voc_xml(k, self.path, os.getcwd(), self.database, self.get_image_dims(),
                                    self.reset_annotation_boxes(), self.labels[k])

                else:
                    if k in self.annotations and len(self.annotations[k]) > 0:
                        self.annotations[k] = self.annotations[k][:-1]
                        # self.remove_zero_annotations()
                    if k in self.annotations and len(self.annotations[k]) % 2 == 1:
                        self.annotations[k].pop()
                        self.labels[k].pop()
                        self.new_box = None
                        save_to_voc_xml(k, self.path, os.getcwd(), self.database, self.get_image_dims(),
                                        self.reset_annotation_boxes(), self.labels[k])
                        self.reset_highlight()
            elif key == 'T':  # 'T' to toggle the labels on or off
                self.labels_on = not self.labels_on
            elif str(key).isnumeric():
                self.highlighted = False
                self.change_selected_label(key)
            elif key == 'Q':
                k = self.paths[self.iter]
                if int(len(self.annotations[k])) > 1:
                    self.highlighted = True
                    self.selected_annot -= 1
                    self.selected_annot = (int(len(self.annotations[k]) / 2) + self.selected_annot) % int(len(self.annotations[k]) / 2)
            elif key == 'E':
                k = self.paths[self.iter]
                if int(len(self.annotations[k])) > 1:
                    self.highlighted = True
                    self.selected_annot += 1
                    self.selected_annot = self.selected_annot % int(len(self.annotations[k]) / 2)


if __name__ == '__main__':
    snappy_annotator = App()
    snappy_annotator.run()
