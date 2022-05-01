
def data_split(input_dataset, train_path, val_path, test_path, train_split, val_split):
    """
    Split dataset into training, validation and test directories 
    Args:
        input_dataset (string): path to dataset
        train_split (float): proportion of training test size in dataset
        val_split (float): proportion of validation test size in training set
    Returns:
        datasets: (list): 3 tuples, each with information required to organize all image paths into training, validation and test data.
    """
    original_path_list = list(paths.list_images(input_dataset))
    
    random.seed(7)
    random.shuffle(original_path)

    index = int(len(original_path_list) * train_split)
    train_path = original_path_list[:index]
    test_path = original_path_list[index:]

    index = int(len(train_path) * val_split)
    val_path = train_path[:index]
    train_path = train_path[index:]

    datasets = [("training", train_path, train_path),
                ("validation", val_path, val_path),
                ("test", test_path, test_path)]


    for (set_type, original_path, base_path) in datasets:
        print(f'Building {set_type} set')

        if not os.path.exists(base_path):
            print(f'Building directory {base_path}')
            os.makedirs(base_path)

        for path in original_path:
            file = path.split(os.path.sep)[-1]
            label = file[-5:-4]

            label_path = os.path.sep.join([base_path, label])
            if not os.path.exists(label_path):
                print(f'Building directory {label_path}')
                os.makedirs(label_path)

            new_path = os.path.sep.join([label_path, file])
            shutil.copy2(path, new_path)

    return datasets
