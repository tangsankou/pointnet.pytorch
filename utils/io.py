import os
import torch
import torch.nn as nn

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
        

def save_model(model, args, name):
   #  if len(args.gpu_id) == 1:
    torch.save(model.state_dict(), 'checkpoints_cls/%s/models/' % (args.exp_name)+name+'.t7')
    """ else:
        torch.save(model.module.state_dict(), 'checkpoints_cls/%s/models/' % (args.exp_name)+name+'.t7') """
        
def load_model(args, classifier):
    model_path = args.model
    """ if model_path[-1] == ']': #MC models
        model_path = model_path[1:-1].split(',')
        model_path = ''.join([s for s in model_path[level] if s not in [' ', '\'', '\"']]) """
    assert os.path.isfile(model_path), '\'{}\' model file does not exist.'.format(model_path)
    classifier.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    """ if len(cfg.DEVICES.GPU_ID) > 1:
        model = nn.DataParallel(model, device_ids=cfg.DEVICES.GPU_ID) """
    return classifier
