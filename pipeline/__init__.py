import importlib
import torch.utils.data as data
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

def create_pipeline(pipeline_type, kwargs):
    '''
        create pipeline by given options
    '''
    pipeline_class = find_pipeline_using_name(pipeline_type)
    pipeline = pipeline_class(**kwargs)
    print(f"pipeline [{type(pipeline).__name__}] was created successfully")
    return pipeline


def find_pipeline_using_name(pipeline_type):
    pipeline_filename = 'pipeline.' + pipeline_type + 'Pipeline'
    pipelinelib = importlib.import_module(pipeline_filename)

    pipeline = None    
    target_pipeline_name = pipeline_type + 'Pipeline'
    for name, cls in pipelinelib.__dict__.items():
        if name.lower() == target_pipeline_name.lower() and issubclass(cls, DiffusionPipeline):
            pipeline = cls
    if pipeline is None:
        raise NotImplementedError("In %s.py, there should be a subclass of DiffusionPipeline with class name that matches %s in lowercase." % (pipeline_filename, target_pipeline_name))

    return pipeline