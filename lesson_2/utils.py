from fastai import *
from fastai.vision.all import *
from fastcore.all import *
import requests
import os
from azure.cognitiveservices.search.imagesearch import ImageSearchClient
from msrest.authentication import CognitiveServicesCredentials
from pathlib import Path
import time

class bingImagesRetriever:
    def __init__(self, apiKeyName=None):
        self.apiKeyName = apiKeyName
        self.apiKey = None
    
    def retrieveApiKey(self):
        
        self.apiKey = os.environ.get(self.apiKeyName)
    
    def setApiKeyManually(self, apiKey):
        self.apiKey = apiKey
        
    def obtainImagesUrlsBing(self, term, minSz=128, maxImages=150):
        
        params = {'q':term, 'count': maxImages, 'min_height': minSz, 'min_width':minSz}
        headers = {"Ocp-Apim-Subscription-Key":self.apiKey}
        searchUrl = "https://api.bing.microsoft.com/v7.0/images/search"
        response = requests.get(searchUrl, headers=headers, params=params)
        response.raise_for_status()
        searchResults = response.json()    
        
        return L(searchResults.get('value'))
    
    def downloadImagesToPath(self, generalName, specificNamesArray, pathName):
        path = Path(pathName)
        if not path.exists():
            path.mkdir()
        
        for term in specificNamesArray:
            dest = (path/term)
            dest.mkdir(exist_ok=True)
            searchTerm = '{} {}'.format(term, generalName)
            print('Searching for {}'.format(searchTerm))
            urls = self.obtainImagesUrlsBing(searchTerm)
            download_images(dest=dest, urls=urls.attrgot('contentUrl'))
            print('Done! Now we will be waiting for 10 seconds to not get an error from exceeding the call limit')
            time.sleep(10)
            
    def unlinkFailed(self, pathName):
        fns = get_image_files(pathName)
        failed = verify_images(fns)
        failed.map(Path.unlink);
        print('Bad downloaded images unlinked!')