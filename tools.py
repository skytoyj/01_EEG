from models import *
from opts import get_opts

def pick_models(opts, data_num=512):
    if opts.denoise_network == 'SimpleCNN':
        model = SimpleCNN(data_num).to(opts.device)
                     
    elif opts.denoise_network == 'FCNN':  
        model = FCNN(data_num).to(opts.device)
                
    elif opts.denoise_network == 'ResCNN':
        model = ResCNN(data_num).to(opts.device)

    elif opts.denoise_network == 'CTNet':
        if data_num == 1024:
            model = CTNet(data_num, drop_rate=0.3).to(opts.device)
        else:
            model = CTNet(data_num).to(opts.device)
    
    elif opts.denoise_network == 'NovelCNN':
        model = NovelCNN(data_num).to(opts.device) 
        
    else:
        print("model name is error!")
        pass
    return model
    