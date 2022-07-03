def get_transform(istrain):
    base_size = 520
    crop_size = 480

    min_size = int((0.5 if istrain else 1.0) * base_size)
    max_size = int((2.0 if istrain else 1.0) * base_size)
    transforms = list()
    transforms.append(T.RandomResize(min_size, max_size))
    if istrain:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))   # new version
        transforms.append(T.RandomCrop(crop_size))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def get_transform_v2(istrain):
    base_size = 480
    crop_size = 480

    min_size = int((0.5 if istrain else 1.0) * base_size)
    max_size = int((2.0 if istrain else 1.0) * base_size)
    transforms = list()
    if istrain:
        transforms.append(T.RandomResize(min_size, max_size))
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))   # new version
        transforms.append(T.RandomCrop(crop_size))
    else:
        transforms.append(T.Resize((base_size, base_size)))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize_V2())
    return T.Compose(transforms)


def get_transform_v3(istrain):
    base_size = 480
    crop_size = 480

    # min_size = int((0.5 if istrain else 1.0) * base_size)
    # max_size = int((2.0 if istrain else 1.0) * base_size)
    transforms = list()
    if istrain:
        transforms.append(T.RandomAffine())
        # transforms.append(T.Resize((base_size, base_size)))
        # transforms.append(T.RandomResize(min_size, max_size))
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))   # new version
        transforms.append(T.RandomCrop(crop_size))
    else:
        transforms.append(T.Resize((base_size, base_size)))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


def get_transform_v4(istrain):
    base_size = 480
    crop_size = 480

    min_size = int((0.5 if istrain else 1.0) * base_size)
    max_size = int((2.0 if istrain else 1.0) * base_size)
    transforms = list()
    if istrain:
        transforms.append(T.RandomResize(min_size, max_size))
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))   # new version
        transforms.append(T.RandomCrop(crop_size))
    else:
        transforms.append(T.Resize((base_size, base_size)))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)