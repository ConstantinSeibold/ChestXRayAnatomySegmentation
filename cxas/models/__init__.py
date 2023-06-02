import gdown, os, torch
from cxas.label_mapper import id2label_dict

model_urls = {
    'UNet_ResNet50_default': 'https://drive.google.com/file/d/1Y9zubvMzkYHoAqz-NvV6vniH5FKAF2iV/view?usp=drive_link'
}

def get_model(model_name, gpus=''):
    assert model_name.split('_')[0] in list(model_getter.keys())
    
    model = model_getter[model_name.split('_')[0]](model_name)
    download_weights(model_name)
    model = load_weights(model, model_name)
    
    gpus = [int(i) for i in gpus.split(',') if len(i)>0]
    
    if len(gpus) > 1:
        assert (torch.cuda.is_available())
        model.to(gpus[0])
        model = torch.nn.DataParallel(model,device_ids =gpus).cuda()
    elif len(gpus)==1:
        assert (torch.cuda.is_available())
        model.to(gpus[0])
        
    return model


def get_unet(model_name):
    from .UNet.backbone_unet import BackboneUNet
    return BackboneUNet(model_name, len(id2label_dict.keys()))

def download_weights(model_name:str)->None:
    os.makedirs('./weights/', exist_ok=True)
    out_path = './weights/{}'.format(model_name+'.pth')
    if os.path.isfile(out_path):
        return
    else:
        gdown.download(model_urls[model_name], out_path, quiet=False, fuzzy=True)
        return

def load_weights(model, model_name:str):
    out_path = './weights/{}'.format(model_name+'.pth')
    assert os.path.isfile(out_path)
    
    checkpoint = torch.load(out_path, map_location='cuda:0')
    # import pdb; pdb.set_trace()
    if 'module' in list(checkpoint['model'].keys())[0] :
        for i in list(checkpoint['model'].keys()):
            checkpoint['model'][i[len('module.'):]] = checkpoint['model'].pop(i)
    model.load_state_dict(checkpoint['model'], strict = False)
    return model

model_getter = {
    'UNet': get_unet,
}