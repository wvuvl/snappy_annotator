"""
Methods for saving annotations to Pascal VOC format
"""

import os
import colored

error_msg = colored.fg("red") + colored.attr("bold")


# Takes annotation and other data for current image and translates into Pascal VOC format.
def save_to_voc(filename, folder, path, database, dims, annotations, labels):
    p = filename.find('.')
    print(filename[:p])

    with open(os.path.join(folder, filename[:p] + '_annotations.xml'), "w") as x:

        # Write metadata
        x.write('<annotation>\n')
        x.write('\t<folder>{}</folder>\n'.format(folder))
        x.write('\t<filename>{}</filename>\n'.format(filename))
        x.write('\t<path>{}</path>\n'.format(os.path.join(path, folder, filename)))
        x.write('\t<source>\n\t\t<database>{}</database>\n\t</source>\n'.format(database))
        x.write('\t<size>\n\t\t<width>{}</width>\n\t\t<height>{}</height>\n\t\t<depth>{}</depth>\n\t</size>\n'
                .format(dims[1], dims[0], dims[2]))
        x.write('\t<segmented>0</segmented>\n')

        # Write annotations
        for i in range(0, int(len(annotations)/2)):
            x.write('\t<object>\n')
            x.write('\t\t<name>{}</name>\n'.format(labels[i]))
            x.write('\t\t<pose>Unspecified</pose>\n')
            x.write('\t\t<truncated>0</truncated>\n')
            x.write('\t\t<difficult>0</difficult>\n')
            x.write('\t\t<bndbox>\n')
            x.write('\t\t\t<xmin>{}</xmin>\n'.format(annotations[i*2][0]))
            x.write('\t\t\t<ymin>{}</ymin>\n'.format(annotations[i*2][1]))
            x.write('\t\t\t<xmax>{}</xmax>\n'.format(annotations[i*2+1][0]))
            x.write('\t\t\t<ymax>{}</ymax>\n'.format(annotations[i*2+1][1]))
            x.write('\t\t</bndbox>\n')
            x.write('\t</object>\n')

        x.write('</annotation>')


# Reads in an xml file, pulls all important information and returns it to program in usable data format
# NOTE: For metadata, currently only reads folder, filename, path, database, and size,
# and for each annotation, only reads label name and bounding box dimensions.
# NOTE: This method will return the path, database, and image dimension metadata, but currently does nothing
# with it. When saving, this information will be overwritten by the save function
def load_from_voc(file_pth, filename):
    path = ''
    database = ''
    dims = tuple()
    annotation = []
    labels = []
    filename = os.path.join(file_pth, str(filename[:filename.find('.')]) + '_annotations.xml')

    if os.path.exists(filename):
        with open(filename, "r") as x:
            line = x.readline().strip()

            # Metadata block - assuming metadata is at beginning of file?
            while line != '<object>' and line != '</annotation>':
                if line.startswith('<path>'):
                    path = line[6:-7]
                if line.startswith('<database>'):
                    database = line[10:-11]
                if line.startswith('<size>'):
                    w, h, d = 0, 0, 0
                    line = x.readline().strip()
                    while line != '</size>':
                        if line.startswith('<width>'):
                            w = int(line[7:-8])
                        elif line.startswith('<height>'):
                            h = int(line[8:-9])
                        elif line.startswith('<depth>'):
                            d = int(line[7:-8])
                        line = x.readline().strip()
                    dims = (w, h, d)  # Note that the image dimensions are returned in w, h, d order
                line = x.readline().strip()

            # Annotation loop
            while line == '<object>':
                while line != '</object>':

                    if line.startswith('<name>'):
                        labels.append(line[6:-7])

                    if line.startswith('<bndbox>'):
                        xmin, ymin, xmax, ymax = -1, -1, -1, -1
                        line = x.readline().strip()
                        while line != '</bndbox>':
                            if line.startswith('<xmin>'):
                                xmin = int(line[6:-7])
                            if line.startswith('<ymin>'):
                                ymin = int(line[6:-7])
                            if line.startswith('<xmax>'):
                                xmax = int(line[6:-7])
                            if line.startswith('<ymax>'):
                                ymax = int(line[6:-7])
                            line = x.readline().strip()
                        # If one or more dimensions are not read:
                        if xmin == -1 or ymin == -1 or xmax == -1 or ymax == -1:
                            print(colored.stylize("WARNING: The current file ({}) is corrupted. One or more dimensions"
                                                  " of a bounding box are not included.".format(filename), error_msg))
                        annotation.append((xmin, ymin))
                        annotation.append((xmax, ymax))

                    line = x.readline().strip()
                line = x.readline().strip()
    return path, database, dims, annotation, labels
