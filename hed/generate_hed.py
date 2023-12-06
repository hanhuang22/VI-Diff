import os
import glob
from PIL import Image
from controlnet_aux import HEDdetector

if __name__ == '__main__':
    hed = HEDdetector.from_pretrained('./' ,'ControlNetHED.pth')

    folder = 'trainA'
    target_path = f'outputs/{folder}'
    os.makedirs(target_path, exist_ok=True)
    test_path = f'./datasets/RegDB/{folder}/'
    image_list = glob.glob(test_path + "*.bmp")
    image_list = sorted(image_list)
    nimgs = len(image_list)
    print("totally {} images".format(nimgs))
    for i in range(nimgs):
        img = Image.open(image_list[i]).resize((64, 128))
        fn, ext = os.path.splitext(image_list[i])
        fn = fn.split('/')[-1]
        img_hed = hed(img, detect_resolution=64, image_resolution=64)
        img_hed = img_hed.convert('L')
        img_hed.save(os.path.join(target_path, f"{fn}.png"))
        print("Saving to '" + os.path.join(target_path, image_list[i][0:-4]) +
              "', Processing %d of %d..." % (i + 1, nimgs))