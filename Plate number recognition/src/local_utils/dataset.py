import os
import random
import xml.etree.ElementTree as ET
from glob import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.local_utils.constants import (
    VALIDATION_SPLIT,
    RANDOM_SEED,
)


def clean_dataset(images_path: str, annotations_path: str):
    print('Cleaning dataset')

    images = os.listdir(images_path)
    annotations = os.listdir(annotations_path)

    valid_images = []

    print('\tVerifying annotations')
    for annotation in annotations:
        img4annotation = f'{annotation[:-4]}.jpg'

        if img4annotation in images:
            valid_images.append(img4annotation)

            print(f'\t\tAnnotation "{annotation}" for "{img4annotation}" verified')
        else:
            os.remove(os.path.join(annotations_path, annotation))

            print(f'\t\tAnnotation "{annotation}" for "{img4annotation}" removed')

    print('\tRemoving non-valid images')
    for image in images:
        if image not in valid_images:
            os.remove(os.path.join(images_path, image))

            print(f'\t\tImage "{image}" removed')

    max_filename_len = len(str(len(valid_images)))
    print('\tFilling the gaps')
    for i in range(len(valid_images)):
        filename = valid_images[i][:-4]
        if filename != str(i).zfill(max_filename_len):
            os.rename(os.path.join(images_path, valid_images[i]),
                      os.path.join(images_path, f'{str(i).zfill(max_filename_len)}.jpg'))

            print(f'\t\tImage "{valid_images[i]}" renamed to "{str(i).zfill(max_filename_len)}.jpg"')

            os.rename(os.path.join(annotations_path, f'{filename}.xml'),
                      os.path.join(annotations_path, f'{str(i).zfill(max_filename_len)}.xml'))

            print(f'\t\tAnnotation "{filename}.xml" renamed to "{str(i).zfill(max_filename_len)}.xml"')

    print('\n\tDone')


def split_dataset(images_path: str, annotations_path: str, training_set_file: str, validation_set_file: str):
    print('Splitting dataset to training & validation sets')

    images = os.listdir(images_path)
    annotations = os.listdir(annotations_path)

    dataset_size = len(images)
    validation_size = int(dataset_size * VALIDATION_SPLIT)

    print(f'\tDataset size: {dataset_size}')
    print(f'\tUsing validation split = {VALIDATION_SPLIT * 100}%')
    print(f'\t\tTraining set size: {dataset_size - validation_size}')
    print(f'\t\tValidation set size: {validation_size}')

    random.seed(RANDOM_SEED)
    validation_indices = sorted(random.sample(range(dataset_size), validation_size))

    training_set = [f'{images[i]} {annotations[i]}' for i in list(range(dataset_size)) if i not in validation_indices]
    validation_set = [f'{images[i]} {annotations[i]}' for i in validation_indices]

    print('\tTraining set:')
    for elem in training_set:
        print(f'\t\t{elem}')

    print('\tValidation set:')
    for elem in validation_set:
        print(f'\t\t{elem}')

    with open(training_set_file, mode='w') as file:
        file.write('\n'.join(training_set))

    with open(validation_set_file, mode='w') as file:
        file.write('\n'.join(validation_set))

    print('\n\tDone')


def get_labels(annotations_path: str, labels_file: str):
    print('Labels:')

    classnames = set()

    for xml in glob(os.path.join(annotations_path, '*.xml')):
        tree = ET.parse(xml)
        root = tree.getroot()

        for elem in root.findall('object'):
            try:
                classname = elem.find('name').text
                classnames.add(classname)
            except TypeError as e:
                print(xml)
                print(e)

    classnames = sorted(classnames)

    for classname in classnames:
        print(f'\t{classname.upper()}')

    with open(labels_file, mode='w') as file:
        file.write('\n'.join(sorted(classnames)))

    print('\n\tDone')
