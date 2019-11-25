# -*- coding: utf-8 -*-
"""
CNN/flair_segmentation
"""
from context import Context
from deepmedic import DeepMedic
from unet import UNet
from vnet import VNet
from loss import RegularizedCrossEntropyLoss, CrossEntropyLoss
from trainer import Trainer
from saver import Saver
from batch_manager import DeepMedicBatchManager
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt


def train(network_name='unet', nb_of_epochs=50):

    context = None

    if network_name == 'unet':

        network = UNet(input_dimensions=np.array([56, 56, 56, 1]),
                       nb_of_classes=2,
                       depth=2)

        batch_manager = DeepMedicBatchManager(input_tile_dimensions=network.input_dimensions,
                                              output_tile_dimensions=network.output_dimensions[:-1],
                                              training_directory='D:\untitled\ku\new',
                                              nb_of_training_images=24,
                                              nb_of_training_images_to_load=18,
                                              nb_of_tiles_per_training_image=1000,
                                              nb_of_tiles_per_training_batch=10,
                                              nb_of_testing_images=6,
                                              nb_of_tiles_per_testing_image=1000,
                                              nb_of_tiles_per_testing_batch=20,
                                              tile_foreground_to_background_ratio=0.2)

        trainer = Trainer(initial_learning_rate=0.0005,
                          loss_function=CrossEntropyLoss())

        context = Context(network=network,
                          batch_manager=batch_manager,
                          trainer=trainer,
                          nb_of_training_tiles_before_evaluation=1800)

    elif network_name == 'vnet':

        network = VNet(input_dimensions=np.array([64, 64, 64, 1]),
                       nb_of_classes=2,
                       depth=4)

        batch_manager = DeepMedicBatchManager(input_tile_dimensions=network.input_dimensions,
                                              output_tile_dimensions=network.output_dimensions[:-1],
                                              training_directory='D:\untitled\ku\new',
                                              nb_of_training_images=24,
                                              nb_of_training_images_to_load=18,
                                              nb_of_tiles_per_training_image=1000,
                                              nb_of_tiles_per_training_batch=10,
                                              nb_of_testing_images=6,
                                              nb_of_tiles_per_testing_image=1000,
                                              nb_of_tiles_per_testing_batch=20,
                                              tile_foreground_to_background_ratio=0.2)

        trainer = Trainer(initial_learning_rate=0.001,
                          loss_function=CrossEntropyLoss())

        context = Context(network=network,
                          batch_manager=batch_manager,
                          trainer=trainer,
                          nb_of_training_tiles_before_evaluation=1800)

    elif network_name == 'deepmedic':

        network = DeepMedic(input_dimensions=np.array([57, 57, 57, 1]),
                            sampling_factor=3,
                            nb_of_classes=2)

        batch_manager = DeepMedicBatchManager(input_tile_dimensions=network.input_dimensions,
                                              output_tile_dimensions=network.output_dimensions[:-1],
                                              training_directory='D:\untitled\ku\new',
                                              nb_of_training_images=24,
                                              nb_of_training_images_to_load=18,
                                              nb_of_tiles_per_training_image=1000,
                                              nb_of_tiles_per_training_batch=200,  # 20 works well, let's see for 40
                                              nb_of_testing_images=6,
                                              nb_of_tiles_per_testing_image=1000,
                                              nb_of_tiles_per_testing_batch=400,
                                              tile_foreground_to_background_ratio=0.2)

        trainer = Trainer(initial_learning_rate=0.0005,  # 0.00001 works well, but bn allows for much higher learning_rates, so to be checked...
                          loss_function=CrossEntropyLoss())

        context = Context(network=network,
                          batch_manager=batch_manager,
                          trainer=trainer,
                          nb_of_training_tiles_before_evaluation=1800)

    context.build_graph()
    context.train(nb_of_epochs)


def predict(directory, image_ids):

    batch_manager = DeepMedicBatchManager(input_tile_dimensions=np.array([56, 56, 56, 1]),
                                          output_tile_dimensions=np.array([16, 16, 16]),
                                          nb_of_tiles_per_prediction_batch=40)

    #batch_manager = DeepMedicBatchManager(input_tile_dimensions=np.array([57, 57, 57, 1]),
    #                                      output_tile_dimensions=np.array([9, 9, 9]),
    #                                      nb_of_tiles_per_prediction_batch=400)

    context = Context(batch_manager=batch_manager)

    for i in image_ids:
        image = np.load(directory + '/' + (i < 100) * '0' + (i < 10) * '0' + str(i) + '.npz')['arr_0']
        segmented_image = context.predict(image)
        np.savez_compressed(directory + '/segmentation/' + (i < 100) * '0' + (i < 10) * '0' + str(i) + '.npz', segmented_image)
        sitk.WriteImage(sitk.GetImageFromArray(segmented_image), directory + '/segmentation/' + (i < 100) * '0' + (i < 10) * '0' + str(i) + '.mha')


def main():

    #train('unet', 1000)
    predict('D:\untitled\ku\new', [2, 3, 4, 8, 13, 20])


main()