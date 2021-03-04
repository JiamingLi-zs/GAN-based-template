"""Create datasets for training and testing."""
import csv
import os
import random
import IC_datasets

def create_list(foldername, fulldir=True, suffix=".jpg"):
    """
    :param foldername: The full path of the folder.
    :param fulldir: Whether to return the full path or not.
    :param suffix: Filter by suffix.

    :return: The list of filenames in the folder with given suffix.

    """
    file_list_tmp = os.listdir(foldername)
    file_list = []
    label_list = []
    if fulldir:
        for item in file_list_tmp:
            if item.endswith(suffix):
                file_list.append(os.path.join(foldername, item))
                flie_name, _ = item.split('.')
                if int(flie_name) > 554:
                    label = 1
                else:
                    label = 0
                label_list.append(label)
    else:
        for item in file_list_tmp:
            if item.endswith(suffix):
                flie_name, _ = item.split('.')
                if int(flie_name) > 554:
                    label = 1
                else:
                    label = 0
                label_list.append(label)
    return file_list, label_list

def create_dataset(data_name='test_data', do_shuffle=True, do_select=False, select_sign='1'):

    image_path_a= IC_datasets.PATH_TO_data[data_name]
    list_a, list_b= create_list(image_path_a, True, '.jpg')
    num_rows = IC_datasets.DATASET_TO_SIZES[data_name]
    all_data_tuples = []
    if do_select == True:
        for i in range(num_rows):
            if list_b[i] == select_sign:
                all_data_tuples.append((list_a[i], list_b[i]))
        output_path = IC_datasets.PATH_TO_CSV[select_sign]
    else:
        for i in range(num_rows):
            all_data_tuples.append((
                list_a[i % len(list_a)],
                list_b[i % len(list_b)]))
        output_path = IC_datasets.PATH_TO_CSV[data_name]
    if do_shuffle is True:
        random.shuffle(all_data_tuples)
    with open(output_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for data_tuple in enumerate(all_data_tuples):
            csv_writer.writerow(list(data_tuple[1]))

if __name__ == '__main__':
    create_dataset(data_name='train_data', do_shuffle=True, do_select=False, select_sign='0')
