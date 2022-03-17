import torch.nn as nn
import torchvision
from models.bert import BertClassifier, BertFeaturizer, DistilBertClassifier
from models.resnet_multispectral import ResNet18
from models.layers import Identity
from models.gnn import GINVirtual

def initialize_model(config, d_out):
    if config.model == 'resnet18_ms':
        # multispectral resnet 18
        model = ResNet18(num_classes=d_out, **config.model_kwargs)
    elif config.model in ('resnet50', 'resnet34', 'wideresnet50', 'densenet121'):
        model = initialize_torchvision_model(
            name=config.model,
            d_out=d_out,
            **config.model_kwargs)
    elif 'bert' in config.model:
        if d_out is None:
            model = BertFeaturizer.from_pretrained(config.model, **config.model_kwargs)
        elif 'distil' in config.model:
            model = DistilBertClassifier.from_pretrained(
                config.model,
                num_labels=d_out,
                **config.model_kwargs)

            trainable_layers = [model.distilbert.transformer.layer[-1], model.pre_classifier, model.classifier]
        else:
            model = BertClassifier.from_pretrained(
                config.model,
                num_labels=d_out,
                **config.model_kwargs)

            trainable_layers = [model.bert.encoder.layer[-1], model.bert.pooler, model.classifier]
            # For DP training, freeze embedding layer
            for name, param in model.named_parameters():
                if name.startswith('bert.embeddings'):
                    param.requires_grad = False

        if config.freeze_bert:
            for p in model.parameters():
                p.requires_grad = False

            for layer in trainable_layers:
                for p in layer.parameters():
                    p.requires_grad = True


    elif config.model == 'logistic_regression':
        model = nn.Linear(out_features=d_out, **config.model_kwargs)
    elif config.model == 'gin-virtual':
        model = GINVirtual(num_tasks=d_out, **config.model_kwargs)
    else:
        raise ValueError('Model not recognized.')
    return model

def initialize_torchvision_model(name, d_out, **kwargs):
    # get constructor and last layer names
    if name == 'wideresnet50':
        constructor_name = 'wide_resnet50_2'
        last_layer_name = 'fc'
    elif name == 'densenet121':
        constructor_name = name
        last_layer_name = 'classifier'
    elif name in ('resnet50', 'resnet34'):
        constructor_name = name
        last_layer_name = 'fc'
    else:
        raise ValueError(f'Torchvision model {name} not recognized')
    # construct the default model, which has the default last layer
    constructor = getattr(torchvision.models, constructor_name)
    model = constructor(**kwargs)
    # adjust the last layer
    d = getattr(model, last_layer_name).in_features
    if d_out is None:  # want to initialize a featurizer model
        last_layer = Identity(d)
        model.d_out = d
    else:  # want to initialize a classifier for a particular num_classes
        last_layer = nn.Linear(d, d_out)
        model.d_out = d_out
    setattr(model, last_layer_name, last_layer)
    # set the feature dimension as an attribute for convenience
    return model
