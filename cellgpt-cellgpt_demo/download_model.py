from pycrawlers import huggingface
#import os
#os.environ["HF_ENDPOINT"] = "http://zhoujieli.tech/huggingface"
# 实例化类
hg = huggingface()
url= 'https://huggingface.co/Salesforce/codet5p-110m-embedding/tree/main'
path = '/root/code_model'
hg.get_data(url, path)
