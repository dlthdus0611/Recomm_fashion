# import library
import matplotlib
matplotlib.use('tkagg')
import skimage
from io import BytesIO
from deepfashion2 import DeepFashion2Config
import mrcnn.model as modellib
from urllib import request
from PIL import Image
import pandas as pd

# load model
config = DeepFashion2Config()
config.GPU_COUNT=1
config.BATCH_SIZE=1

MODEL_DIR = '../../PycharmProjects/Recomm_fashion/final_weights.h5'
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(MODEL_DIR, by_name=True)

# musinsa image lists
musinsa = pd.read_csv('stylebook.csv')

# define function cropping and saving images
def save_cropped(model, image_url, cat_name, index):
    # print("Running on {}".format(args.image))

    res = request.urlopen(image_url).read()
    image = skimage.io.imread(BytesIO(res))
    r = model.detect([image], verbose=1)[0]

    class_names = [None, "short_sleeved_shirt", "long_sleeved_shirt", "short_sleeved_outwear",
                   "long_sleeved_outwear", "vest", "sling", "shorts", "trousers", "skirt",
                   "short_sleeved_dress", "long_sleeved_dress", "vest_dress", "sling_dress"]
    boxes = r['rois']
    class_ids = r['class_ids']

    cat = []; path = []
    for i in range(len(boxes)):

        y1, x1, y2, x2 = boxes[i]
        class_id = class_ids[i]; label = class_names[class_id]

        cropped_image = Image.fromarray(image[y1:y2, x1:x2])
        image_path = './musinsa_cropped/{}_index{}_{}.jpg'.format(cat_name, index, i)
        cropped_image.save(image_path)

        cat.append(label); path.append(image_path)

    return cat, path

cropped_musinsa = []
for n_row in range(0,len(musinsa)):
    print('index_{} --ing'.format(n_row))
    row = musinsa.loc[n_row]
    try:
        cat_lst, path_lst = save_cropped(model, row['image_url'], row['category_name'], row['index'])
        for num in range(len(cat_lst)):
            cropped_musinsa.append((row['category_name'], row['index'], cat_lst[num], path_lst[num]))
    except:
        continue

cropped_musinsa_df = pd.DataFrame(cropped_musinsa, columns=['category_name', 'index', 'label', 'image_path'])
cropped_musinsa_df.to_csv('cropped_musinsa.csv', index=False, encoding='cp949')




