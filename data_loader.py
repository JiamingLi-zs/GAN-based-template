import tensorflow as tf
import IC_datasets



def _load_samples(csv_name, image_type):
    filename_queue = tf.train.string_input_producer(
        [csv_name])

    reader = tf.TextLineReader()
    _, csv_filename = reader.read(filename_queue)

    record_defaults = [tf.constant([], dtype=tf.string),
                       tf.constant([], dtype=tf.int32)]

    image, label = tf.decode_csv(
        csv_filename, record_defaults=record_defaults)

    file_contents = tf.read_file(image)
    if image_type == '.jpg':
        image_decoded = tf.image.decode_jpeg(
            file_contents, channels=3)
    elif image_type == '.png':
        image_decoded = tf.image.decode_png(
            file_contents, channels=3, dtype=tf.uint8)

    return image_decoded, label


def load_data(dataset_name, do_shuffle=True, one_hot=True, batch_size=IC_datasets.batch_size):
    """
    :param one_hot: whether make the label become one_hot
    :param dataset_name: The name of the dataset.
    :param do_shuffle: Shuffle switch.
    :return: inputs['image'], inputs['label']
    """
    if dataset_name not in IC_datasets.DATASET_TO_SIZES:
        raise ValueError('split name %s was not recognized.'
                         % dataset_name)

    csv_name = IC_datasets.PATH_TO_CSV[dataset_name]

    image, label = _load_samples(csv_name, IC_datasets.DATASET_TO_IMAGETYPE[dataset_name])
    inputs = {'image': image, 'label': label}

    # image Preprocessing:
    inputs['image'] = tf.image.resize_images(inputs['image'], [84, 16])

    #注意：一定要把样本的形状固定 [64, 16, 3]，在批处理的时候要求所有数据形状必须定义
    inputs['image'].set_shape([84, 16, 3])

    inputs['image'] = tf.subtract(tf.div(inputs['image'], 127.5), 1)

    if one_hot is True:
        # label Preprocessing:
        inputs['label']=tf.one_hot(inputs['label'], 2)

    # Batch
    if do_shuffle is True:
        inputs['image'], inputs['label'] = tf.train.shuffle_batch(
            [inputs['image'], inputs['label']], batch_size, 5000, 100)
    else:
        inputs['image'], inputs['label'] = tf.train.batch(
            [inputs['image'], inputs['label']], batch_size)

    return inputs
