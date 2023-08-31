from bardapi import BardCookies
from typing import Any, List
from io import BytesIO
from functools import cached_property
import numpy as np
from PIL import Image
import cv2
import json
import re

class Preprocess:
    default = {
        "resize": {
            "size":500,
            "scale": True
        },
        "background": {},
        "grey": {},
        "square_crop": {
            "size": 255
        }
    }
    @staticmethod
    def resize(image, size, scale=True):
        h,w = image.size
        scale_factor = min(size/h, size/w)
        return image.resize((int(h*scale_factor), int(w*scale_factor)))
    
    @staticmethod
    def background(image, **kwargs):
        # Convert PIL Image to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Create a mask of the same size as the image, filled with 0s
        mask = np.zeros(image_cv.shape[:2], np.uint8)

        # Create foreground and background models
        fgModel = np.zeros((1, 65), np.float64)
        bgModel = np.zeros((1, 65), np.float64)

        # Define a rectangle around the subject (assuming the subject is in the center of the image)
        rect = (50, 50, image_cv.shape[1]-100, image_cv.shape[0]-100)

        # Apply GrabCut algorithm
        cv2.grabCut(image_cv, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

        # Create a mask where sure and likely backgrounds are set to 0, otherwise 1
        mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')

        # Multiply image with new mask to subtract background
        image_cv = image_cv*mask2[:,:,np.newaxis]

        # Change all black (also shades of blacks) pixels to white
        image_cv[mask2 == 0] = [255, 255, 255]

        # Convert image back to PIL format
        return Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    
    @staticmethod
    def grey(image, **kwargs):
        return image.convert("L")
    
    @staticmethod
    def square_crop(image, size):
        # Get the original dimensions of the image
        width, height = image.size

        # Calculate the dimensions for center cropping
        crop_size = min(width, height)
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size

        # Create a new image with the desired size
        cropped_image = Image.new("RGB", (size, size))

        # Calculate the coordinates to paste the original image onto the new image
        paste_left = (size - crop_size) // 2
        paste_top = (size - crop_size) // 2
        paste_right = paste_left + crop_size
        paste_bottom = paste_top + crop_size

        # Paste the center cropped region onto the new image
        cropped_image.paste(image.crop((left, top, right, bottom)), (paste_left, paste_top))

        return cropped_image
    
    @staticmethod
    def apply(image, ops_args: List):
        """
        ops_args is a list of tuples (operation : override dict)
        """
        im = image
        for op, args in ops_args:
            d = dict(Preprocess.default[op])
            d.update(args)
            im = getattr(Preprocess, op)(im, **d)
            
        return im
    

class Clothing: 
    # TODO: define preprocessing as a list of functors
    def __init__(self, image, preprocessing) -> None:
        self.image = Preprocess.apply(image, preprocessing)
        self.vec = None

    # def _vectorize(embed_engine):
    #     return embed_engine(self.image)

    @cached_property
    def bytes(self):
        with BytesIO() as b:
            self.image.save(b, format="PNG")
            return b.getvalue()


class AllClothes:
    def __init__(self) -> None:
        self.clothes = []
        self.embeddings = []

    def __getitem__(self, index):
        return self.clothes[index]
    
class ImageCaptioner():
    # we are going to use: https://github.com/dsdanielpark/Bard-API
    def __init__(self, cookie_dict) -> None:
        self.bard = BardCookies(cookie_dict=cookie_dict)
        self.prompt = """Describe this article of clothing. You should describe the article of categories in a json format using the example below:
```
{
    color: COLOR,
    category: Pants, Tshirt, Jacket, Polo Shirt, ...,
    description: SHORT DESCRIPTION,
    material: MATERIAL,
    season, WINTER, SUMMER, FALL, SPRING, ALL
    
}
```
"""
        self.pattern = r"\{(.*?)\}"
    def extract(self, content):
        return json.loads(f"{{\n{re.findall(self.pattern, content, re.DOTALL)[0]}\n}}") # the fstring here just adds {} to the string cuz the regex removes it
    
    def exec(self, clothing: Clothing):
        response = self.bard.ask_about_image(self.prompt, clothing.bytes)
        try:
            return self.extract(response['content'])
        except:
            return response['content']

    def __call__(self, clothing: Clothing) -> Any:
        return self.exec(clothing)