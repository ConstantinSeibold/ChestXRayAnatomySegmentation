import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from .file_io import FileLoader, FileSaver, get_folder_loader
from .models import get_model
from .extraction import Extractor

class CXAS(nn.Module):
    def __init__(self, 
                 model_name:str='UNet_ResNet50_default', 
                 gpus:str=''):
        """
        Create Chest X-Ray anatomy segmentation model

        Parameters
        ----------
            model_name: which model/weight to load, if weights are not stored, they will be downloaded at '~/weights/'
            
            gpus: on which gpu to perform inference on

        """
        super(CXAS,self).__init__()
        self.model = get_model(model_name, gpus)
        self.gpus = gpus
        self.fileloader = FileLoader(gpus)
        self.filesaver  = FileSaver()
        self.extractor = Extractor()
        self.eval()
            
    def process_file(self, 
                     filename: str, 
                     do_store:bool=False, 
                     output_directory:str='./',
                     create:bool = False,
                     storage_type:str='npy'
                    ) -> np.array:
        """
        Create segmentation of image file, can store predictions in desired directory in desired format

        Parameters
        ----------
            filename: path of file to process, currently supported types [.dcm, .jpg, .png]
            do_store: bool indicating whether to store prediction
            output_directory: desired path of output directory
            storage_type: desired type to store segmentation prediction as, currently supported types [dicom-seg, jpg, png, npy, npz, json]

        Returns
        -------
            prediction: model output dictionary containing [feats: network features , logits: unnormalized network logit scores, data: input data, segmentation_preds: thresholded multi-label segmentations] 
        """
        assert os.path.isfile(filename)
        
        if not create:
            assert os.path.isdir(output_directory)
        else:
            os.makedirs(output_directory, exist_ok=True)
            
        file_dict = self.fileloader.load_file(filename)
        file_dict['filename'] = [file_dict['filename']]
        file_dict['file_size'] = [file_dict['file_size']]
        with torch.no_grad():
            predictions = self.model(file_dict)
            
        if do_store:
            self.store_prediction(predictions, output_directory, storage_type)
        return predictions
    
    def process_folder(self, 
                       input_directory_name: str, 
                       output_directory:str, 
                       storage_type:str = 'npy', 
                       create:bool = False, 
                       batch_size:int=1
                      ) -> None:
        """
        Create segmentations for all image files in directory, stores predictions in desired output directory in desired format

        Parameters
        ----------
            input_directory_name: path of file to process, currently supported types [.dcm, .jpg, .png]
            output_directory: desired path of output directory
            storage_type: desired type to store segmentation prediction as, currently supported types [dicom-seg, jpg, png, npy, npz, json]
            create: whether to create the output directory
            batch_size: batch size used for the forward passes of the model
        """
        assert os.path.isdir(input_directory_name)
        if not create:
            assert os.path.isdir(output_directory)
        else:
            os.makedirs(output_directory, exist_ok=True)
            
        dataloader = get_folder_loader(input_directory_name, self.gpus, batch_size, )
        
        if storage_type == 'json':
            from .io_utils.create_annotations import get_coco_json_format, \
                create_category_annotation
            from .io_utils.mask_to_coco import binary_mask_to_rle, toBox, mask_to_annotation
            from .label_mapper import id2label_dict, category_ids
            import json

            coco_format = get_coco_json_format()
            coco_format["categories"] = create_category_annotation(category_ids)
            coco_format["images"] = []
            coco_format["annotations"] = []
            img_id = 1
            base_ann_id = 1
            
        
        for file_dict in tqdm(dataloader):
            if len(self.gpus)>0:
                file_dict['data'] = file_dict['data'].to('cuda:{}'.format(self.gpus[0]))
                
            with torch.no_grad():
                predictions = self.model(file_dict)
                
            if storage_type == 'json':
                for i in range(len(predictions['filename'])):
                    mask = self.resize_to_numpy(
                        segmentation = predictions['segmentation_preds'][i], 
                        file_size = predictions['file_size'][i]
                        )
                    annotations = mask_to_annotation(
                                                    mask = mask, 
                                                    base_ann_id = base_ann_id, 
                                                    img_id = img_id
                                                )
                    base_ann_id += len(annotations)
                    coco_format["images"]      += [{'id':img_id, 'file_name': predictions['filename'][i]}]
                    coco_format["annotations"] += annotations
                    img_id += 1
            else:
                self.store_prediction(predictions, output_directory, storage_type)
        
        if storage_type == 'json':
            os.makedirs(output_directory,exist_ok=True)
            out_path = os.path.join(output_directory, input_directory_name.split('/')[-1]+'.json')
            with open(out_path,"w") as outfile:
                json.dump(coco_format, outfile)
            
    def store_prediction(self, 
                         predictions: dict, 
                         output_directory:str, 
                         storage_type:str) -> None:
        """
        Store all elements in batch
        
        Parameters
        ----------
            predictions: model output dictionary containing [feats: network features , logits: unnormalized network logit scores, data: input data, segmentation_preds: thresholded multi-label segmentations] 
            output_directory: desired path of output directory
            storage_type: desired type to store segmentation prediction as, currently supported types [dicom-seg, jpg, png, npy, npz, json]
            
        """
        
        for i in range(len(predictions['filename'])):
            pred = self.resize_to_numpy(
                 segmentation = predictions['segmentation_preds'][i], 
                 file_size = predictions['file_size'][i]
                )
            self.filesaver.save_prediction(pred, output_directory, predictions['filename'][i], storage_type)
    
    def resize_to_numpy(self, 
                        segmentation: torch.Tensor, 
                        file_size, 
                       ) -> torch.Tensor:
        """
        Resize binary torch prediction mask to desired size
        
        Parameters
        ----------
            segmentation: model output dictionary containing [feats: network features , logits: unnormalized network logit scores, data: input data, segmentation_preds: thresholded multi-label segmentations] 
            file_size: desired path of output directory            
        """
        pred = segmentation.float()
        pred = F.interpolate(pred.unsqueeze(0), file_size, mode='nearest')      
        pred = pred[0].bool().to('cpu').numpy()
        return pred 
        
    def extract_features_for_file(self, 
                                  filename: str,
                                  feat_to_extract: str,
                                  draw:bool = False,
                                  create:bool = False,
                                  do_store:bool=False, 
                                  output_directory:str='./',
                                  storage_type:str='npy'
                                 ) -> dict:
        """
        Create segmentation of image file and extract features in relation to the segmentation. Can store predictions in desired output directory in desired format.

        Parameters
        ----------
            filename: path of file to process, currently supported types [.dcm, .jpg, .png]
            feat_to_extract:  which features to extract in relation to the segmentation
            draw: draw the origin of the features
            create: whether to create the output directory
            do_store: bool indicating whether to store prediction
            output_directory: desired path of output directory
            storage_type: desired type to store segmentation prediction as, currently supported types [dicom-seg, jpg, png, npy, npz, json]

        Returns
        -------
            features: extracted feature score and if so designated its visualization
        """
        assert os.path.isfile(filename)        
        
        if not create:
            assert os.path.isdir(output_directory)
        else:
            os.makedirs(output_directory, exist_ok=True)
            
        predictions = self.process_file(
                    filename,
                    do_store = do_store, 
                    output_directory = output_directory,
                    storage_type = storage_type,
                )

        feat_dict = self.extractor.extract(
            file = predictions['segmentation_preds'][0].cpu().bool().numpy(),
            method = feat_to_extract,
            draw=draw,
        )
        
        print('The {} for the file {} is '.format(feat_to_extract, filename.split('/')[-1]),feat_dict['score'])
        
        if do_store:
            scores = [{'score':self.extractor.extract(
                                        file = predictions['segmentation_preds'][0].cpu().bool().numpy(),
                                        method = feat_to_extract,
                                        draw=False,
                                    )['score'],
                                'filename': predictions['filename'][0],
                               }]
            pd.DataFrame(scores).to_csv(os.path.join(output_directory, filename.split('/')[-1].split('.')[0]+'.csv'))
        
        return feat_dict
    
    def extract_features_for_folder(self, 
                                    input_directory_name: str, 
                                    output_directory:str, 
                                    feat_to_extract: str, 
                                    create:bool = False, 
                                    store_pred: bool = False,
                                    storage_type:str = 'npy',
                                    batch_size:int=1
                                   ) -> None:
        """
        Create segmentation of image file and extract features in relation to the segmentation. Can store predictions in desired output directory in desired format.

        Parameters
        ----------
            input_directory_name: path of file to process, currently supported types [.dcm, .jpg, .png]
            output_directory: desired path of output directory
            storage_type: desired type to store segmentation prediction as, currently supported types [dicom-seg, jpg, png, npy, npz, json]
            create: whether to create the output directory
            batch_size: batch size used for the forward passes of the model
            feat_to_extract:  which features to extract in relation to the segmentation
            draw: draw the origin of the features
            create: whether to create the output directory
            store_pred: bool indicating whether to store prediction

        """
        assert os.path.isdir(input_directory_name)
        if not create:
            assert os.path.isdir(output_directory)
        else:
            os.makedirs(output_directory, exist_ok=True)
            
        
        scores = []
        
        dataloader = get_folder_loader(input_directory_name, self.gpus, batch_size, )
        
        if (storage_type == 'json') and store_pred:
            from cxas.io_utils.create_annotations import get_coco_json_format, \
                create_category_annotation
            from cxas.io_utils.mask_to_coco import binary_mask_to_rle, toBox, mask_to_annotation
            from cxaslabel_mapper import id2label_dict, category_ids
            import json

            coco_format = get_coco_json_format()
            coco_format["categories"] = create_category_annotation(category_ids)
            coco_format["images"] = []
            coco_format["annotations"] = []
            img_id = 1
            base_ann_id = 1
            
        for file_dict in tqdm(dataloader):
            if len(self.gpus)>0:
                file_dict['data'] = file_dict['data'].to('cuda:{}'.format(self.gpus[0]))
                
            with torch.no_grad():
                predictions = self.model(file_dict)
                
            for i in range(len(predictions['segmentation_preds'])):
                scores += [{'score':self.extractor.extract(
                                    file = predictions['segmentation_preds'][i].cpu().bool().numpy(),
                                    method = feat_to_extract,
                                    draw=False,
                                )['score'],
                            'filename': predictions['filename'][i],
                           }]
            
            if store_pred:    
                if storage_type == 'json':
                    for i in range(len(predictions['filename'])):
                        mask = self.resize_to_numpy(
                            segmentation = predictions['segmentation_preds'][i], 
                            file_size = predictions['file_size'][i]
                            )
                        annotations = mask_to_annotation(
                                                        mask = mask, 
                                                        base_ann_id = base_ann_id, 
                                                        img_id = img_id
                                                    )
                        base_ann_id += len(annotations)
                        coco_format["images"]      += [{'id':img_id, 'file_name': predictions['filename'][i]}]
                        coco_format["annotations"] += annotations
                        img_id += 1
                else:
                    self.store_prediction(predictions, output_directory, storage_type)

        if (storage_type == 'json') and store_pred:
            os.makedirs(output_directory,exist_ok=True)
            out_path = os.path.join(output_directory, input_directory_name.split('/')[-1]+'.json')
            with open(out_path,"w") as outfile:
                json.dump(coco_format, outfile)

        pd.DataFrame(scores).to_csv(os.path.join(output_directory, input_directory_name.split('/')[-1]+'.csv'))
        
    
    def forward(self, image_batch) -> dict:
        """
        """
        if type(image_batch) is not dict:
            image_batch = {'data': image_batch}
        return self.model(image_batch)