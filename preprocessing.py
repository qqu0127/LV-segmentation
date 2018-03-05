import dicom, cv2, re
import os, fnmatch, sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from itertools import izip
from utils import center_crop, lr_poly_decay, get_SAX_SERIES

SAX_SERIES = get_SAX_SERIES()
SUNNYBROOK_ROOT_PATH = '../Medical-Image-Analysis/Data/'
TEST_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'Sunnybrook Cardiac MR Database ContoursPart1',
                            'OnlineDataContours')
TEST_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH, 'challenge_online/challenge_online')
VAL_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'Sunnybrook Cardiac MR Database ContoursPart2',
                            'ValidationDataContours')
VAL_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH, 'challenge_validation')

TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            'Sunnybrook Cardiac MR Database ContoursPart3',
                            'TrainingDataContours')
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        'challenge_training')

def shrink_case(case):
    toks = case.split('-')
    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x
    return '-'.join([shrink_if_number(t) for t in toks])


class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r'/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-.*', ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))
    
    def __str__(self):
        return '<Contour for case %s, image %d>' % (self.case, self.img_no)
    
    __repr__ = __str__


def read_contour(contour, data_path):
    filename = 'IM-%s-%04d.dcm' % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(data_path, contour.case, filename)
    f = dicom.read_file(full_path)
    img = f.pixel_array.astype('int')
    mask = np.zeros_like(img, dtype='uint8')
    coords = np.loadtxt(contour.ctr_path, delimiter=' ').astype('int')
    cv2.fillPoly(mask, [coords], 1)
    if img.ndim < 3:
        img = img[..., np.newaxis]
        mask = mask[..., np.newaxis]
    
    return img, mask


def map_all_contours(contour_path, shuffle=False):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files,
                        'IM-0001-*-icontour-manual.txt')]
    if shuffle:
        print('Shuffling data')
        np.random.shuffle(contours)
    print('Number of examples: {:d}'.format(len(contours)))
    contours = map(Contour, contours)
    
    return contours


def export_all_contours(contours, data_path, crop_size):
    print('\nProcessing {:d} images and labels ...\n'.format(len(contours)))
    images = np.zeros((len(contours), crop_size, crop_size, 1))
    masks = np.zeros((len(contours), crop_size, crop_size, 1))
    for idx, contour in enumerate(contours):
        img, mask = read_contour(contour, data_path)
        img = center_crop(img, crop_size=crop_size)
        mask = center_crop(mask, crop_size=crop_size)
        images[idx] = img
        masks[idx] = mask

    return images, masks

def prepareDataset(contour_path, img_path, crop_size):
    contours = map_all_contours(contour_path)
    img, mask = export_all_contours(contours,
                                            img_path,  
                                            crop_size=crop_size)
    return img, mask
    