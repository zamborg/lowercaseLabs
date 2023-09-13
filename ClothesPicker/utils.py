from typing import Any, List, Dict, Iterable
from bardapi import BardCookies
from io import BytesIO
from functools import cached_property
import numpy as np
from PIL import Image
import cv2
import json
import re
from collections import OrderedDict

import os
import psycopg2
from dotenv import load_dotenv
from typing import Callable
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

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
   

class CALLABLE:
    def exec(self, **kwargs):
        raise NotImplementedError("Must override `exec` method")
    
    def __call__(self, **kwargs):
        return self.exec(**kwargs)

class Clothing: 
    def __init__(self, color, category, description, material, season, fp, preprocessing = [], vector = None, **kwargs) -> None:
        self.color, self.category, self.material = color, category, material
        self.description, self.season = description, season
        self.image = Preprocess.apply(Image.open(fp), preprocessing)
        self.vector = vector
        self.json = kwargs.get("json", False) #return all content as a json in __str__

    @staticmethod
    def _from_DB(db_tuple, pre=[]):
        _, col, cat, desc, mat, sea, fp = db_tuple
        return Clothing(col,cat,desc,mat,sea,fp,pre)

    @cached_property
    def bytes(self):
        with BytesIO() as b:
            self.image.save(b, format="PNG")
            return b.getvalue()

    def __str__(self) -> str:
        if self.json:
            data = {
                "color": self.color,
                "category": self.category,
                "material": self.material,
                "description": self.description,
                "season": self.season
            }
            return json.dumps(data)
        return self.description


class AllClothes:
    def __init__(self, clothes = []) -> None:
        self.clothes = clothes

    def __getitem__(self, index):
        return self.clothes[index]
    
    def apply(self, func: callable):
        """
        applies func on all clothes in self.clothes
        func should be inplace if mutating
        """
        for c in self.clothes:
            func(c)
        return
    
    @staticmethod
    def _from_DB(pre=[]):
        load_dotenv()
        db = psycopg2.connect(os.environ['DBURL'])
        with db.cursor() as cur:
            data = cur.execute("SELECT * FROM clothing")
            data = cur.fetchall()
            cur.close()
        db.close()
        return AllClothes(clothes=[Clothing._from_DB(d, pre) for d in data])
    

class ImageCaptioner(CALLABLE):
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
        

class DB(CALLABLE):
    def __init__(self) -> None:
        load_dotenv()
        self.DB = psycopg2.connect(os.environ['DBURL'])
        self.DB.set_session(autocommit=True)
        self.count = 0

    def _reconnect(self):
        del self.DB
        self.DB = psycopg2.connect(os.environ['DBURL'])
        self.count = 0

    def exec(self, sql: str):
        if self.count == 5:
            self._reconnect()
        with self.DB.cursor() as cursor:
            self.count += 1
            return cursor.execute(sql)
        
class VectorDB():
    def __init__(self, unit_vectors=True) -> None:
        """
        Custom implementation of a vectorDB for search.
        ## TODO: maybe change this to use the .vector property of Clothing?
        ## TODO: consider 
        """
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.tokenizer_kwargs = {"padding":True, "truncation":True, "return_tensors":'pt'}
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.embeddings = np.zeros((0,384))# zero array to vstack
        self.itoo = dict() # empty dict for index to object lookup
        self.unit_vectors = unit_vectors

    def insert(self, obj, access_func: Callable = lambda x : str(x)):
        """
        This function vectorizes the object using access_func 
            stores the object's index in the self.itoo lookup dictionary
        INPUT:
            object: the object to insert
            access_func: method to get a string from the object for vectorization (default __str__)
        """
        vector = self.embed(obj, access_func)
        self.itoo[len(self.embeddings)] = obj
        self.embeddings = np.vstack([self.embeddings, vector.numpy()])
        return
    
    def batch_insert(self, obj: Iterable, access_func: Callable = lambda x : str(x)):
        """
        Just a vectorized pass of the objects.callable to the model for vectorization
        exact same as insert
        
        Accepts A SINGLE ACCESS FUNC -- no heterogeneous access.

        inserts ALL objects into self
        """
        tokens = []
        for o in obj:
            # insert all objects
            self.itoo[len(self.itoo)] = o
            tokens.append(access_func(o))
        tokens = self.tokenizer(tokens, **self.tokenizer_kwargs)

        with torch.no_grad():
            vectors = self.model(**tokens).last_hidden_state
        vectors = vectors.mean(dim=1)# NxD
        if self.unit_vectors:
            vectors = F.normalize(vectors, 2, dim=1)
        self.embeddings = np.vstack([self.embeddings, vectors.numpy()])
        return

    def embed(self, obj, access_func: Callable = lambda x : str(x)):
        """
        returns vector embedding of object
        callable takes object as input and returns a string -- default: str(object)

        returns torch tensor vector with avg pooling
        """
        string = access_func(obj)
        with torch.no_grad():
            vector = self.model(**self.tokenizer(string, **self.tokenizer_kwargs)).last_hidden_state # model is a transformer so just pass the entire input unrolled

        vector = vector.mean(dim=1).squeeze() # this is avg pooling -- maybe should be max_pooling?
        if self.unit_vectors:
            vector = F.normalize(vector, 2, 0)
        return vector
    
    def knn(self, obj, access_func: Callable = lambda x : str(x), k=1):
        """
        returns the k nearest neighbors to obj using access_func to extract a string
        searches against the vector db
        returns an ordered list of [(obj, score)]
        """
        vector = self.embed(obj, access_func).numpy() # D dimensional vector
        dot_products = np.dot(self.embeddings, vector)
        sorted_indices = np.argsort(dot_products)[::-1][:k]
        sorted_scores = dot_products[sorted_indices][:k]
        ordered_list = [(self.itoo[idx], score) for idx, score in zip(sorted_indices, sorted_scores)]
        return ordered_list
        
    def __getitem__(self, index):
        return self.itoo[index]
    def __setitem__(self, index, item):
        self.itoo[index] = item
    def __len__(self):
        return len(self.itoo)
    def pop(self, index):
        self.itoo.pop(index)
        self.embeddings = np.delete(self.embeddings, index, 0) # delete the index elemtent in the 0th embedding 