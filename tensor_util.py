import functools

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

class tf_read:
    def __init__(self):
        pass
    def fromcsv(file_path, **kwargs):
        """"
        Read csv data with filepath
        """
        dataset = tf.data.experimental.make_csv_dataset(
                        file_path,
                        batch_size=5, # Artificially small to make examples easier to show.
                        na_value="NULL",
                        num_epochs=1,
                        ignore_errors=True, 
                        num_parallel_reads=20,
                        **kwargs)
        return dataset
    def countcsv_rows(path_to_file):
        total_row = 0
        for filename in os.listdir(path_to_file):
            n_row = sum(1 for row in open(os.path.join(path_to_file,filename)))
            total_row += n_row
            print(f'{filename} has {n_row} rows')
        print(f'Total has {total_row} rows')

class tf_frame:
    def __init__(self):
        pass
    def show_batch(dataset):
        for batch in dataset.take(1):
            for key,value in batch.items():
                print("{:20s}: {}".format(key,value.numpy()))

    def split_dataset(dataset: tf.data.Dataset, validation_data_fraction: float):
        """
        Splits a dataset of type tf.data.Dataset into a training and validation dataset using given ratio. Fractions are
        rounded up to two decimal places.
        @param dataset: the input dataset to split.
        @param validation_data_fraction: the fraction of the validation data as a float between 0 and 1.
        @return: a tuple of two tf.data.Datasets as (training, validation)
        """

        validation_data_percent = round(validation_data_fraction * 100)
        if not (0 <= validation_data_percent <= 100):
            raise ValueError("validation data fraction must be ∈ [0,1]")

        dataset = dataset.enumerate()
        train_dataset = dataset.filter(lambda f, data: f % 100 > validation_data_percent)
        validation_dataset = dataset.filter(lambda f, data: f % 100 <= validation_data_percent)

        # remove enumeration
        train_dataset = train_dataset.map(lambda f, data: data)
        validation_dataset = validation_dataset.map(lambda f, data: data)

        return train_dataset, validation_dataset