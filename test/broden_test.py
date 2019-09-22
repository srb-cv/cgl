import numpy as np
import os
from PIL import Image

def test_broden_image():
    root_dir = "/scratch/data/broden1_227/images/ade20k/"

    im_1 = Image.open(os.path.join(root_dir, "ADE_train_00018774_object.png"))
    im_1_np = np.asarray(im_1)

    print(im_1_np.shape)

    c1 = im_1_np[:, :, 0]
    c2 = im_1_np[:, :, 1]
    c3 = im_1_np[:, :, 2]

    print(c1.shape)

    print(np.unique(c1))
    print(np.unique(c2))
    print(np.unique(c3))

    print(c2)

def test_create_index_file():
    import pandas as pd
    root_dir = "/scratch/data/synthetic_labels_v2/images"
    import glob, os
    os.chdir(root_dir)
    image_list=[]
    color_list=[]
    shape_list=[]
    color_shape_list=[]

    for file in glob.glob("*.jpg"):
        basename = os.path.splitext(file)[0]
        print(basename)
        image_list.append(basename+".jpg")
        color_list.append(basename+"_color.png")
        shape_list.append(basename+"_shape.png")
        color_shape_list.append(basename+"_color_shape.png")


    df = pd.DataFrame({'image':image_list,
                       'split':['train'] * len(image_list),
                       'ih':[64] * len(image_list),
                       'iw':[64] * len(image_list),
                       'sh':[64] * len(image_list),
                       'sw':[64] * len(image_list),
                       'color':color_list,
                        'shape':shape_list,
                        'shape_color':color_shape_list}
                        )

    print(df)
    df.to_csv(r'/scratch/data/synthetic_labels_v2//index.csv',index=False)

def test_create_label_file():
    import pandas as pd
    name_list =['RED', 'GREEN', 'BLUE','CIRCLE', 'SQUARE', 'TRIANGLE', 'CIRCLE_RED','CIRCLE_GREEN','CIRCLE_BLUE',
                'SQUARE_RED', 'SQUARE_GREEN', 'SQUARE_BLUE','TRIANGLE_RED', 'TRIANGLE_GREEN', 'TRIANGLE_BLUE']
    category_list = ['color','color','color','shape','shape','shape','shape_color','shape_color','shape_color',
                     'shape_color','shape_color','shape_color','shape_color','shape_color','shape_color']
    number = np.arange(1,16,1)

    df = pd.DataFrame({'number':number,
                       'name':name_list,
                       'category':category_list
            })

    print(df)
    df.to_csv(r'/scratch/data/synthetic_labels_v2/label.csv', index=False)


def test_create_color_label():
    import pandas as pd
    code_list = [1,2,3]
    number_list = [1,2,3]
    name_list = ['RED','GREEN','BLUE']
    df = pd.DataFrame({'code': code_list,
                       'number': number_list,
                       'name': name_list})
    df.to_csv(r'/scratch/data/synthetic_labels_v2/c_color.csv', index=False)

def test_create_shape_label():
    import pandas as pd
    code_list = [1,2,3]
    number_list = [4,5,6]
    name_list = ['CIRCLE','SQUARE','TRIANGLE']
    df = pd.DataFrame({'code': code_list,
                       'number': number_list,
                       'name': name_list})
    df.to_csv(r'/scratch/data/synthetic_labels_v2/c_shape.csv', index=False)

def test_create_shape_color_label():
    import pandas as pd
    code_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    number_list = [7, 8, 9, 10, 11, 12, 13, 14, 15]
    name_list = ['CIRCLE_RED','CIRCLE_GREEN','CIRCLE_BLUE',
                'SQUARE_RED', 'SQUARE_GREEN', 'SQUARE_BLUE','TRIANGLE_RED', 'TRIANGLE_GREEN', 'TRIANGLE_BLUE']
    df = pd.DataFrame({'code': code_list,
                       'number': number_list,
                       'name': name_list})
    df.to_csv(r'/scratch/data/synthetic_labels_v2/c_shape_color.csv', index=False)

def test_create_category():
    import pandas as pd
    name_list = ['color','shape','shape_color']
    first=np.array([1,4,7])
    last=np.array([3,6,15])
    df = pd.DataFrame({'name':name_list,
                       'first':first,
                       'last':last})
    df.to_csv(r'/scratch/data/synthetic_labels_v2/category.csv', index=False)
    print(df)


test_create_index_file()
test_create_label_file()
test_create_color_label()
test_create_shape_label()
test_create_shape_color_label()
test_create_category()
