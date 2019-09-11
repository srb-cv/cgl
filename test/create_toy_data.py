import random

import numpy as np
from matplotlib import pyplot as plt
import cv2
from itertools import combinations_with_replacement
from itertools import product
import os

class ToyDataset:
    def __init__(self):
        self.colors = ['RED', 'BLUE', 'GREEN']
        self.shapes = ['CIRCLE', 'SQUARE', 'TRIANGLE']
        self.classes = []
        self.color_dict = {'RED': (255, 0, 0), 'GREEN': (0, 255, 0), 'BLUE': (0, 0, 255)}
        self.color_label_dict = {'RED': 10, 'GREEN': 20, 'BLUE': 30}
        self.shape_label_dict = {'CIRCLE': 10, 'SQUARE': 20, 'TRIANGLE': 30}

        self.label_dict = {'CIRCLE_RED': 10, 'CIRCLE_GREEN': 20, 'CIRCLE_BLUE': 30,
                           'SQUARE_RED': 40, 'SQUARE_GREEN': 50, 'SQUARE_BLUE': 60,
                           'TRIANGLE_RED': 70, 'TRIANGLE_GREEN': 80, 'TRIANGLE_BLUE': 90}
        self.shape_size = 12
        self.image_height = 64
        self.image_width = 64
        #self.no_images_per_class = 640
        self.no_images_per_class = 6
        self.root_data_path = "synthetic_data_v1/data"
        self.label_concept_path = "synthetic_data_v1/concept_label"

    def create_classes(self):
        perm_colors = product(self.colors, repeat=2)
        comb_shapes = combinations_with_replacement(self.shapes, 2)
        classes = list(product(perm_colors, comb_shapes))
        classes = self.__remove_unimportant_classes(classes)
        self.classes = classes

        for i, a_class in enumerate(self.classes):
            print(i, a_class)
        print("No of classes in the toy dataset: ", len(self.classes))

    def __remove_unimportant_classes(self, classes):
        classes.remove((('GREEN', 'BLUE'), ('CIRCLE', 'CIRCLE')))
        classes.remove((('GREEN', 'BLUE'), ('SQUARE', 'SQUARE')))
        classes.remove((('GREEN', 'BLUE'), ('TRIANGLE', 'TRIANGLE')))

        classes.remove((('BLUE', 'RED'), ('CIRCLE', 'CIRCLE')))
        classes.remove((('BLUE', 'RED'), ('SQUARE', 'SQUARE')))
        classes.remove((('BLUE', 'RED'), ('TRIANGLE', 'TRIANGLE')))

        classes.remove((('GREEN', 'RED'), ('CIRCLE', 'CIRCLE')))
        classes.remove((('GREEN', 'RED'), ('SQUARE', 'SQUARE')))
        classes.remove((('GREEN', 'RED'), ('TRIANGLE', 'TRIANGLE')))
        return classes

    def draw_shapes(self, img, random_figure, random_color, pts):
        x = pts[0]
        y = pts[1]
        c = self.shape_size // 2
        if random_figure == 'SQUARE':
            img = cv2.rectangle(img, (x - c, y - c), (x + c, y + c), random_color, -1)
        elif random_figure == 'TRIANGLE':
            pt1 = (x - c, y + c)
            pt2 = (x + c, y + c)
            pt3 = (x, y - c)
            triangle_cnt = np.array([pt1, pt2, pt3])
            img = cv2.drawContours(img, [triangle_cnt], 0, random_color, -1)
        elif random_figure == 'CIRCLE':
            img = cv2.circle(img, (x, y), c, random_color, -1)
        else:
            print("Invalid choice of figure(or may be invalid color)")
            return None
        return img

    def draw_labels_shapes(self, shape_label, random_figure, pts):
        x = pts[0]
        y = pts[1]
        c = self.shape_size // 2
        if random_figure == 'SQUARE':
            shape_label = cv2.rectangle(shape_label, (x - c, y - c), (x + c, y + c),
                                        (self.shape_label_dict[random_figure], 0, 0), -1)
        elif random_figure == 'TRIANGLE':
            pt1 = (x - c, y + c)
            pt2 = (x + c, y + c)
            pt3 = (x, y - c)
            triangle_cnt = np.array([pt1, pt2, pt3])
            shape_label = cv2.drawContours(shape_label, [triangle_cnt], 0,
                                           (self.shape_label_dict[random_figure], 0, 0), -1)
        elif random_figure == 'CIRCLE':
            shape_label = cv2.circle(shape_label, (x, y), c,
                                     (self.shape_label_dict[random_figure], 0, 0), -1)
        else:
            print("Invalid choice of figure(or may be invalid color)")
            return None
        return shape_label

    def draw_label_colors(self, color_label, random_figure, random_color, pts):
        x = pts[0]
        y = pts[1]
        c = self.shape_size // 2
        if random_figure == 'SQUARE':
            color_label = cv2.rectangle(color_label, (x - c, y - c), (x + c, y + c),
                                        (self.color_label_dict[random_color], 0, 0), -1)
        elif random_figure == 'TRIANGLE':
            pt1 = (x - c, y + c)
            pt2 = (x + c, y + c)
            pt3 = (x, y - c)
            triangle_cnt = np.array([pt1, pt2, pt3])
            color_label = cv2.drawContours(color_label, [triangle_cnt], 0,
                                           (self.color_label_dict[random_color], 0, 0), -1)
        elif random_figure == 'CIRCLE':
            color_label = cv2.circle(color_label, (x, y), c, (self.color_label_dict[random_color], 0, 0), -1)
        else:
            print("Invalid choice of figure(or may be invalid color)")
            return None
        return color_label

    def draw_labels(self, label, random_figure, random_color, pts):
        x = pts[0]
        y = pts[1]
        c = self.shape_size // 2
        if random_figure == 'SQUARE':
            label = cv2.rectangle(label, (x - c, y - c), (x + c, y + c),
                                  (self.label_dict[random_figure + '_' + random_color], 0, 0), -1)
        elif random_figure == 'TRIANGLE':
            pt1 = (x - c, y + c)
            pt2 = (x + c, y + c)
            pt3 = (x, y - c)
            triangle_cnt = np.array([pt1, pt2, pt3])
            label = cv2.drawContours(label, [triangle_cnt], 0,
                                     (self.label_dict[random_figure + '_' + random_color], 0, 0), -1)
        elif random_figure == 'CIRCLE':
            label = cv2.circle(label, (x, y), c,
                               (self.label_dict[random_figure + '_' + random_color], 0, 0), -1)
        else:
            print("Invalid choice of figure(or may be invalid color)")
            return None
        return label

    def create_class_images(self, class_tuple):
        c = self.shape_size // 2
        img = np.ones((self.image_height, self.image_width, 3), np.uint8) * 255
        shape_label = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        color_label = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        label = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

        colors, shapes = class_tuple

        x1 = np.random.randint(c, img.shape[1] - c, size=1)
        y1 = np.random.randint(c, img.shape[0] / 2 - c, size=1)
        x2 = np.random.randint(c, img.shape[1] - c, size=1)
        y2 = np.random.randint(img.shape[0] / 2 + c + 2, img.shape[0] - c, size=1)

        pts = [(x1,y1), (x2,y2)]
        random.shuffle(pts)

        self.draw_shapes(img, shapes[0], self.color_dict[colors[0]], pts[0])
        self.draw_labels_shapes(shape_label, shapes[0], pts[0])
        self.draw_label_colors(color_label, shapes[0], colors[0], pts[0])
        self.draw_labels(label, shapes[0], colors[0], pts[0])


        self.draw_shapes(img, shapes[1], self.color_dict[colors[1]], pts[1])
        self.draw_labels_shapes(shape_label, shapes[1], pts[1])
        self.draw_label_colors(color_label, shapes[1], colors[1], pts[1])
        self.draw_labels(label, shapes[1], colors[1], pts[1])
        return img, shape_label, color_label, label

    def create_dataset(self):
        self.create_directories()
        print("Directories successfully created, creating images..")
        for class_no,a_class in enumerate(self.classes):
            separator = "_"
            a_class_path = separator.join([j for i in a_class for j in i])
            class_save_path = os.path.join(self.root_data_path, a_class_path)
            for i in range(self.no_images_per_class):
                image, shape_label, color_label, label = self.create_class_images(a_class)
                base_name = f"{class_no * self.no_images_per_class + i + 1:05d}"
                image_path = os.path.join(class_save_path, base_name+'.jpg')
                cv2.imwrite(image_path, image[..., ::-1])
                label_image_path = os.path.join(self.label_concept_path,base_name+'.jpg')
                cv2.imwrite(label_image_path, image[..., ::-1])
                shape_label_path = os.path.join(self.label_concept_path,base_name+'_shape.png')
                cv2.imwrite(shape_label_path, shape_label[..., ::-1])
                color_label_path = os.path.join(self.label_concept_path,base_name+'_color.png')
                cv2.imwrite(color_label_path, color_label[..., ::-1])
                label_path = os.path.join(self.label_concept_path,base_name+'_color_shape.png')
                cv2.imwrite(label_path, label[..., ::-1])

    def create_directories(self):
        if os.path.exists(self.root_data_path) or os.path.exists(self.label_concept_path):
            print("Folder %s or %s already exists", self.root_data_path, self.label_concept_path)
            return
        else:
            os.makedirs(self.root_data_path)
            os.makedirs(self.label_concept_path)
            if len(self.classes) == 0:
                self.create_classes()
            for a_class in self.classes:
                a_class_path = "_".join([j for i in a_class for j in i])
                os.makedirs(os.path.join(self.root_data_path, a_class_path))

data_obj = ToyDataset()
data_obj.create_dataset()
# data_obj.create_classes()
# img = np.ones((data_obj.image_height, data_obj.image_width, 3), np.uint8) * 255
# image, shape_label, color_label, label = data_obj.create_class_images(data_obj.classes[20])
#
# print("the unique image pixels: \n", np.unique(image.reshape(-1, shape_label.shape[2]), axis=0))
# print("the unique shape label pixels: \n", np.unique(shape_label.reshape(-1, shape_label.shape[2]), axis=0))
# print("the unique color label pixels: \n", np.unique(color_label.reshape(-1, shape_label.shape[2]), axis=0))
# print("the unique label pixels: \n", np.unique(label.reshape(-1, shape_label.shape[2]), axis=0))
#
# plt.imshow(img)
# plt.show()
# plt.imshow(shape_label)
# plt.show()
# plt.imshow(color_label)
# plt.show()
# plt.imshow(label)
# plt.show()

# cv2.imwrite('abc/bcg/image.png',image[...,::-1])
# cv2.imwrite('shape_label.png',shape_label[...,::-1])
# cv2.imwrite('color_label.png',color_label[...,::-1])
# cv2.imwrite('label.png',label[...,::-1])
