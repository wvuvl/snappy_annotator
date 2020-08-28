"""
Methods for saving annotations to Pascal VOC format
"""

import os
import colored
import xml.etree.ElementTree as et
from xml.dom import minidom

error_msg = colored.fg("red") + colored.attr("bold")


# Takes annotation and other data for current image and translates into a Pascal VOC-formatted .xml file.
def save_to_voc_xml(filename, folder, path, database, dims, annotations, labels):
    p = filename.find('.')
    with open(os.path.join(folder, filename[:p] + '_annotations.xml'), "w") as x:
        pass

    xml = et.Element('annotation')
    fold = et.SubElement(xml, 'folder')
    fold.text = folder
    et.SubElement(xml, 'filename').text = filename
    et.SubElement(xml, 'path').text = path
    src = et.SubElement(xml, 'source')
    et.SubElement(src, 'database').text = database
    sz = et.SubElement(xml, 'size')
    et.SubElement(sz, 'width').text = str(dims[1])
    et.SubElement(sz, 'height').text = str(dims[0])
    et.SubElement(sz, 'depth').text = str(dims[2])

    for i in range(0, int(len(annotations) / 2)):
        obj = et.SubElement(xml, 'object')
        et.SubElement(obj, 'name').text = labels[i]
        et.SubElement(obj, 'pose').text = 'Unspecified'
        et.SubElement(obj, 'truncated').text = '0'
        et.SubElement(obj, 'difficult').text = '0'
        bndbox = et.SubElement(obj, 'bndbox')
        et.SubElement(bndbox, 'xmin').text = str(annotations[i * 2][0])
        et.SubElement(bndbox, 'ymin').text = str(annotations[i * 2][1])
        et.SubElement(bndbox, 'xmax').text = str(annotations[i * 2 + 1][0])
        et.SubElement(bndbox, 'ymax').text = str(annotations[i * 2 + 1][1])

    rough_string = et.tostring(xml, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_string = reparsed.toprettyxml(indent="\t")

    p = filename.find('.')
    with open(os.path.join(folder, filename[:p] + '_annotations.xml'), 'w') as x:
        x.writelines(pretty_string)


# Reads in an xml file, pulls all important information and returns it to program in usable data format
# NOTE: For metadata, currently only reads in path, image dimensions, and database,
# and for each annotation, only reads label name and bounding box dimensions.
# NOTE: This method will return the path and database metadata, but currently does nothing
# with them. When saving, this information will be overwritten by the save function
def load_from_voc_xml(file_pth, filename):
    path = ''
    database = ''
    annotation = []
    labels = []
    xml_dims = ()
    filename = os.path.join(file_pth, str(filename[:filename.find('.')]) + '_annotations.xml')

    if os.path.exists(filename):
        tree = et.parse(filename)
        root = tree.getroot()
        if root.find('path') is not None:
            path = root.find('path').text
        src = root.find('source')
        if src is not None and src.find('database') is not None:
            database = src.find('database').text
        if root.find('size') is not None:
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            depth = int(size.find('depth').text)
            xml_dims = (width, height, depth)
        for object in root.iter('object'):
            labels.append(object.find('name').text)
            annotation.append(
                (int(object.find('bndbox').find('xmin').text), int(object.find('bndbox').find('ymin').text)))
            annotation.append(
                (int(object.find('bndbox').find('xmax').text), int(object.find('bndbox').find('ymax').text)))

    return path, database, xml_dims, annotation, labels
