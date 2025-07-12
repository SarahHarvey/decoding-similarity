import os
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import pickle

import dsutils


def get_model_activations_llama(modelname, model, tokenizer, prompt, max_new_tokens = 100, saverep = True, filename = ''):
    """
    Process prompts through the given model in batches and return a dictionary
    containing activations for each recorded feature.
    """

    if torch.cuda.is_available():
        model.to("cuda")
    
    feature_outputs = {}
    hook_handles = []
    
    def get_features(name):
        def hook(module, input, output):
            # print(output[0].shape)
            feature_outputs[name] = output[0].cpu().numpy()  # Detach, move to CPU, and convert to NumPy
        return hook
    
    # Identify the layers you want to extract features from
    # You can inspect the model's named modules:
    # for name, module in model.named_modules():
    #     print(name)
    
    # Example: Register hooks for all the decoder layers
    for i in range(model.config.num_hidden_layers):
        layer_name = f"model.layers.{i}"
        layer = model.get_submodule(layer_name)
        handle = layer.register_forward_hook(get_features(layer_name))
        hook_handles.append(handle)
    
    layer_name = f"model.rotary_emb"
    layer = model.get_submodule(layer_name)
    handle = layer.register_forward_hook(get_features(layer_name))
    hook_handles.append(handle)
    
    layer_name = f"model"
    layer = model.get_submodule(layer_name)
    handle = layer.register_forward_hook(get_features(layer_name))
    hook_handles.append(handle)
    
    layer_name = f"model.embed_tokens"
    layer = model.get_submodule(layer_name)
    handle = layer.register_forward_hook(get_features(layer_name))
    hook_handles.append(handle)
    
    layer_name = f"lm_head"
    layer = model.get_submodule(layer_name)
    handle = layer.register_forward_hook(get_features(layer_name))
    hook_handles.append(handle)
    
    # Prepare input
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")
    
    # Perform forward pass
    with torch.no_grad():
        final_output = model.generate(input_ids, max_new_tokens=max_new_tokens, num_return_sequences=1)
    
    
    # Access the extracted features
    print("Extracted Features:")
    for name, features in feature_outputs.items():
        print(f"Layer: {name}, Shape: {features.shape}")
    
    generated_text = tokenizer.decode(final_output[0], skip_special_tokens=True)
    print(generated_text)
    
    # To remove hooks later:
    for handle in hook_handles:
        handle.remove()
        
    activations = feature_outputs
        
    if saverep:
        save_dir = os.getcwd()
        repDict = {}
        repDict[modelname] = activations
        with open(save_dir + '/reps/' + modelname + '_'+ filename + '.pkl', 'wb') as f:
            pickle.dump(repDict, f)

    return activations




def get_model_activations(modelname, weights, image_data, batch_size=32, saverep = True, filename = ''):
    """
    Process images through the given model in batches and return a dictionary
    containing activations for each recorded feature.
    """
    torchvision_avail_models = models.list_models(module=torchvision.models)
    if modelname in torchvision_avail_models:
        if weights == "first":
            weight_enum = torch.hub.load("pytorch/vision", "get_model_weights", name=modelname)
            weights_avail = [weight for weight in weight_enum]
            loaded_weights = weights_avail[0]
        elif weights == 'random':
            loaded_weights = None
    
        model = torch.hub.load('pytorch/vision', modelname, weights=loaded_weights)

    elif modelname == 'dinov2_vitb14':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using {device} for inference')

    model.eval().to(device)
    all_features = {}
    n = len(image_data)

    feature_outputs = {}
    hook_handles = []

    def get_features(name):
        def hook(module, input, output):
            feature_outputs[name] = output.detach()
            # print(name)
        return hook

    for name, module in model.named_children():
        handle = module.register_forward_hook(get_features(name))
        hook_handles.append(handle)
        print(name)
        # print(module)

    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        for name, module in model.encoder.layers.named_children():
            handle = module.register_forward_hook(get_features(name))
            hook_handles.append(handle)
            # print(name)


    if hasattr(model, 'blocks'):
        prefix='block'
        for name, module in model.blocks.named_children():
            full_name = prefix + ('.' if prefix else '') + name
            handle = module.register_forward_hook(get_features(full_name))
            hook_handles.append(handle)
            print(name)


    if hasattr(model, 'features'):
        prefix='feature'
        for name, module in model.features.named_children():
            full_name = prefix + ('.' if prefix else '') + name
            handle = module.register_forward_hook(get_features(full_name))
            hook_handles.append(handle)
            print(name)

    if hasattr(model, 'layers'):
        prefix='layer'
        for name, module in model.layers.named_children():
            full_name = prefix + ('.' if prefix else '') + name
            handle = module.register_forward_hook(get_features(full_name))
            hook_handles.append(handle)
            print(name)


    if hasattr(model, 'trunk_output'):
        for name, module in model.trunk_output.named_children():
            handle = module.register_forward_hook(get_features(name))
            hook_handles.append(handle)
            print(name)

    #     # Register hooks for layers whose name contains a specific string
    # target_substring = 'encoder'
    # for name, module in model._modules.items():
    #     if target_substring in name:
    #         module.register_forward_hook(get_features(name))
    #         print(name)

    # load preprocessing        
    if (modelname in torchvision_avail_models) and hasattr(loaded_weights, 'transforms') and callable(loaded_weights.transforms):
        preprocess = loaded_weights.transforms()
        print("Preprocessing pipeline:")
        print(preprocess)

    else:
        print("This model does not provide a transforms() method for preprocessing.  Using default.")
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(preprocess)

    for i in range(0, n, batch_size):
        # Convert each numpy image to a PIL image and apply preprocessing.
        batch_imgs = image_data[i:i+batch_size]
        batch_tensors = torch.stack([preprocess(Image.fromarray(img)) for img in batch_imgs]).to(device)
        with torch.no_grad():
            out = model(batch_tensors)
        
        if i == 0:
            # Initialize storage for each feature key.
            all_features['pixels'] = []
            for key in feature_outputs.keys():
                all_features[key] = []
        for key, feat in feature_outputs.items():
            all_features[key].append(feat.cpu().numpy())

        # all_features['pixels'].append(batch_imgs)
        all_features['pixels'].append(batch_tensors)

        print(i)

    # all_features['pixels'] = 
    
    # Concatenate results for each key.
    activations = {key: np.concatenate(val, axis=0) for key, val in all_features.items()}

    if saverep:
        save_dir = os.path.dirname(os.getcwd())
        repDict = {}
        repDict[modelname] = activations
        with open(save_dir + '/reps/' + modelname + '_'+ filename + '.pkl', 'wb') as f:
            pickle.dump(repDict, f)

    return activations



def extract_rep_nsd(model: str, image_data, weights: str ) -> npt.NDArray:
    
    available_models = models.list_models(module=torchvision.models)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using {device} for inference')


    if model == "pixels":

        testmodel = nn.Sequential(*[nn.Identity(), dsutils.Flatten()])
        testmodel.eval().to(device)
        
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
        # dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        # M = len(dataset)
        # trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=0)  # set num_workers to 0 if you are running this on a Mac
    

        n = len(image_data)
        batch_size = n

        for i in range(0, n, batch_size):
            # Convert each numpy image to a PIL image and apply preprocessing.
            batch_imgs = image_data[i:i+batch_size]
            batch_tensors = torch.stack([preprocess(Image.fromarray(img)) for img in batch_imgs]).to(device)
            with torch.no_grad():
                # output = torch.nn.functional.softmax(testmodel(batch_tensors), dim=1)
                y1 = testmodel(batch_tensors)

        # with torch.no_grad():
        #     for inputs, _ in trainloader:
        #         y1 = testmodel(inputs)
    
        return y1, nn.Identity()

    else:
    

        if weights == "first":
            weight_enum = torch.hub.load("pytorch/vision", "get_model_weights", name=model)
            weights_avail = [weight for weight in weight_enum]
            testmodel = torch.hub.load('pytorch/vision', model, weights=weights_avail[0])
    
        elif weights == "last":
            weight_enum = torch.hub.load("pytorch/vision", "get_model_weights", name=model)
            weights_avail = [weight for weight in weight_enum]
            testmodel = torch.hub.load('pytorch/vision', model, weights=weights_avail[-1])
            
        elif weights == "default":
            testmodel = torch.hub.load('pytorch/vision', model, pretrained=True)
    
        elif weights == "random":
            testmodel = torch.hub.load('pytorch/vision', model, pretrained=False)
    
        # elif weights == "all":  TO DO
    
        
        testmodel.eval().to(device)
    
        if model == "inception_v3": 
            preprocess = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        else:
            preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    
        # dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        # M = len(dataset)
        # trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=1)  # set num_workers to 0 if you are running this on a Mac
    
        # module1 = list(testmodel.children())[0:-1]
        # module2 = list(testmodel.children())[-1]
    
        # model_1st = nn.Sequential(*[*module1, dsutils.Flatten()])
        # model_2nd = nn.Sequential(*[module2, dsutils.SoftMaxModule() ])
    
        if 'fc' in dir(testmodel):
            testmodel_2nd = testmodel.fc
            testmodel.fc = nn.Identity()
        elif 'classifier' in dir(testmodel):
            testmodel_2nd = testmodel.classifier
            testmodel.classifier = nn.Identity()
        elif 'head' in dir(testmodel):
            testmodel_2nd = testmodel.head
            testmodel.head = nn.Identity()
        elif 'heads' in dir(testmodel):
            testmodel_2nd = testmodel.heads
            testmodel.heads = nn.Identity()
        else:
            raise ValueError(
                    "Last layer has weird name")
    

        n = len(image_data)
        batch_size = n

        for i in range(0, n, batch_size):
            # Convert each numpy image to a PIL image and apply preprocessing.
            batch_imgs = image_data[i:i+batch_size]
            batch_tensors = torch.stack([preprocess(Image.fromarray(img)) for img in batch_imgs]).to(device)
            with torch.no_grad():
                # output = torch.nn.functional.softmax(testmodel(batch_tensors), dim=1)
                y1 = testmodel(batch_tensors)

        # with torch.no_grad():
        #     for inputs, _ in trainloader:
        #         y1 = testmodel(inputs)
        #         # y2 = testmodel_2nd(y1)
    
    
        return y1, testmodel_2nd






def extract_rep_gen(model: str, data_dir, weights: str ) -> npt.NDArray:
    
    available_models = models.list_models(module=torchvision.models)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using {device} for inference')


    if model == "pixels":

        testmodel = nn.Sequential(*[nn.Identity(), dsutils.Flatten()])
        testmodel.eval().to(device)
        
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
        dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        M = len(dataset)
        trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=0)  # set num_workers to 0 if you are running this on a Mac
    
        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = testmodel(inputs)
    
        return y1, nn.Identity()

    else:
    

        if weights == "first":
            weight_enum = torch.hub.load("pytorch/vision", "get_model_weights", name=model)
            weights_avail = [weight for weight in weight_enum]
            testmodel = torch.hub.load('pytorch/vision', model, weights=weights_avail[0])
    
        elif weights == "last":
            weight_enum = torch.hub.load("pytorch/vision", "get_model_weights", name=model)
            weights_avail = [weight for weight in weight_enum]
            testmodel = torch.hub.load('pytorch/vision', model, weights=weights_avail[-1])
            
        elif weights == "default":
            testmodel = torch.hub.load('pytorch/vision', model, pretrained=True)
    
        elif weights == "random":
            testmodel = torch.hub.load('pytorch/vision', model, pretrained=False)
    
        # elif weights == "all":  TO DO
    
        
        testmodel.eval().to(device)
    
        if model == "inception_v3": 
            preprocess = transforms.Compose([
                transforms.Resize(299),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
        else:
            preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    
        dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        M = len(dataset)
        trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=1)  # set num_workers to 0 if you are running this on a Mac
    
        # module1 = list(testmodel.children())[0:-1]
        # module2 = list(testmodel.children())[-1]
    
        # model_1st = nn.Sequential(*[*module1, dsutils.Flatten()])
        # model_2nd = nn.Sequential(*[module2, dsutils.SoftMaxModule() ])
    
        if 'fc' in dir(testmodel):
            testmodel_2nd = testmodel.fc
            testmodel.fc = nn.Identity()
        elif 'classifier' in dir(testmodel):
            testmodel_2nd = testmodel.classifier
            testmodel.classifier = nn.Identity()
        elif 'head' in dir(testmodel):
            testmodel_2nd = testmodel.head
            testmodel.head = nn.Identity()
        elif 'heads' in dir(testmodel):
            testmodel_2nd = testmodel.heads
            testmodel.heads = nn.Identity()
        else:
            raise ValueError(
                    "Last layer has weird name")
    
        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = testmodel(inputs)
                # y2 = testmodel_2nd(y1)
    
    
        return y1, testmodel_2nd


def extract_rep(model: str, data_dir, weights: str) -> npt.NDArray:

    available_models = ["alexnet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "vgg16", "inceptionv3"]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using {device} for inference')

    if model == "resnet18":
        resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        resnet18.eval().to(device)

        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        M = len(dataset)
        trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=0)  # set num_workers to 0 if you are running this on a Mac

        module1 = list(resnet18.children())[0:9]
        module2 = list(resnet18.children())[9:]

        rnet18_1st = nn.Sequential(*[*module1, dsutils.Flatten()])
        rnet18_2nd = nn.Sequential(*[*module2, dsutils.SoftMaxModule() ])

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = rnet18_1st(inputs)
                # y2 = rnet18_2nd(y1) 

        return y1, rnet18_2nd


    if model == "resnet50":

        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        resnet50.eval().to(device)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        M = len(dataset)
        trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=0)  # set num_workers to 0 if you are running this on a Mac

        module1 = list(resnet50.children())[:9]
        module2 = list(resnet50.children())[9:]

        rnet50_1st = nn.Sequential(*[*module1, dsutils.Flatten()])
        rnet50_2nd = nn.Sequential(*[*module2, dsutils.SoftMaxModule() ])

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = rnet50_1st(inputs)
                # y2 = rnet50_2nd(y1) 

        return y1, rnet50_2nd
    

    if model == "alexnet":

        alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        alexnet.eval().to(device)
        
        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        M = len(dataset)
        trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=0)  # set num_workers to 0 if you are running this on a Mac

        module1 = list(alexnet.children())[:1]
        module2 = list(alexnet.children())[1:2]
        module3 = list(alexnet.children())[2:]

        anet_1st = nn.Sequential(*[*module1, *module2, dsutils.Flatten(), module3[0][0:6]] )
        anet_2nd = nn.Sequential(*[module3[0][6:], dsutils.SoftMaxModule() ])

        with torch.no_grad():
            for inputs, _ in trainloader:
                x1 = anet_1st(inputs)
                # x2 = anet_2nd(x1) 
                
        return x1, anet_2nd

    if model == "resnet34":
 
        resnet34 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        resnet34.eval().to(device)


        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        M = len(dataset)
        trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=0)  # set num_workers to 0 if you are running this on a Mac

        module1 = list(resnet34.children())[0:9]
        module2 = list(resnet34.children())[9:]

        rnet34_1st = nn.Sequential(*[*module1, dsutils.Flatten()])
        rnet34_2nd = nn.Sequential(*[*module2, dsutils.SoftMaxModule() ])

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = rnet34_1st(inputs)
                # y2 = rnet34_2nd(y1) 

        return y1, rnet34_2nd

    if model == "resnet101":

        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        M = len(dataset)
        trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=0)  # set num_workers to 0 if you are running this on a Mac

        resnet101 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        resnet101.eval().to(device)

        module1 = list(resnet101.children())[0:9]
        module2 = list(resnet101.children())[9:]

        rnet101_1st = nn.Sequential(*[*module1, dsutils.Flatten()])
        rnet101_2nd = nn.Sequential(*[*module2, dsutils.SoftMaxModule() ])

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = rnet101_1st(inputs)
                # y2 = rnet101_2nd(y1) 

        return y1, rnet101_2nd

    if model == "resnet152":

        resnet152 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        resnet152.eval().to(device)

        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        M = len(dataset)
        trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=0)  # set num_workers to 0 if you are running this on a Mac

        module1 = list(resnet152.children())[0:9]
        module2 = list(resnet152.children())[9:]

        rnet152_1st = nn.Sequential(*[*module1, dsutils.Flatten()])
        rnet152_2nd = nn.Sequential(*[*module2, dsutils.SoftMaxModule() ])

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = rnet152_1st(inputs)
                # y2 = rnet152_2nd(y1)
                
        return y1, rnet152_2nd

    if model.startswith("vgg"): #model == "vgg16":

        vggX = torch.hub.load('pytorch/vision:v0.10.0', model, pretrained=True)
        vggX.eval().to(device)

        preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        M = len(dataset)
        trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=0)  # set num_workers to 0 if you are running this on a Mac

        module1 = list(vggX.children())[0:2]
        module2 = list(vggX.children())[2:]

        vggX_1st = nn.Sequential(*[*module1, dsutils.Flatten(), module2[0][0:6]] )
        vggX_2nd = nn.Sequential(*[module2[0][6:], dsutils.SoftMaxModule() ])

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = vggX_1st(inputs)
                # y2 = vgg16_2nd(y1) 

        return y1, vggX_2nd


    if model == "inceptionv3":

        inceptionv3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
        inceptionv3.eval().to(device)

        preprocess = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        M = len(dataset)
        trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=0)  # set num_workers to 0 if you are running this on a Mac

        inception_2nd = inceptionv3.fc
        inceptionv3.fc = nn.Identity()

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = inceptionv3(inputs)

        return y1, inception_2nd  


    if model == "squeezenetv1":

        squeezenetv1 = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_0', pretrained=True)
        squeezenetv1.eval().to(device)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        M = len(dataset)
        trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=0)  # set num_workers to 0 if you are running this on a Mac

        squeezenet_2nd = squeezenetv1.classifier
        squeezenetv1.classifier = nn.Flatten()

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = squeezenetv1(inputs)

        return y1, squeezenet_2nd  # TO DO 


    if model == "densenet":

        densenet = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
        densenet.eval().to(device)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        M = len(dataset)
        trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=0)  # set num_workers to 0 if you are running this on a Mac

        densenet_2nd = densenet.classifier
        densenet.classifier = nn.Identity()

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = densenet(inputs)

        return y1, densenet_2nd # TO DO 


    if model == "mobilenetv2":

        mobilenet = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        mobilenet.eval().to(device)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = datasets.ImageFolder(data_dir, transform=preprocess)
        M = len(dataset)
        trainloader = DataLoader(dataset, batch_size=M, shuffle=False, num_workers=0)  # set num_workers to 0 if you are running this on a Mac

        mobilenet_2nd = mobilenet.classifier
        mobilenet.classifier = nn.Identity()

        with torch.no_grad():
            for inputs, _ in trainloader:
                y1 = mobilenet(inputs)

        return y1, mobilenet_2nd  # TO DO 
 

    else: 
        return print("Model name not found.  Available models are: " + ", " .join([model_name for model_name in available_models]))

