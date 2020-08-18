from EmoPy.src.fermodel import FERModel
from EmoPy.src.csv_data_loader import CSVDataLoader
from EmoPy.src.data_generator import DataGenerator
from EmoPy.src.neuralnets import ConvolutionalNN

from pkg_resources import resource_filename,resource_exists

validation_split = 0.15

target_dimensions = (64, 64)
channels = 1
verbose = True

print('--------------- Convolutional Model -------------------')
print('Loading data...')
csv_path = resource_filename('EmoPy.examples','image_data/sample.csv')
target_emotion_map = {0:'happy', 1:'disgust', 2: 'surprise', 3:'sad', 4:'angry', 5:'fear', 6:'neutral'}
data_loader = CSVDataLoader(datapath=csv_path, target_emotion_map = target_emotion_map, image_dimensions = target_dimensions , csv_label_col = 0, csv_image_col = 1, validation_split=validation_split)
dataset = data_loader.load_data()

if verbose:
    dataset.print_data_details()

print('Preparing training/testing data...')
train_images, train_labels = dataset.get_training_data()
train_gen = DataGenerator().fit(train_images, train_labels)
test_images, test_labels = dataset.get_test_data()
test_gen = DataGenerator().fit(test_images, test_labels)

print('Training net...')
model = ConvolutionalNN(target_dimensions, channels, dataset.get_emotion_index_map(), verbose=True)
model.fit_generator(train_gen.generate(target_dimensions, batch_size=5),
                    test_gen.generate(target_dimensions, batch_size=5),
                    epochs=5)

# Save model configuration
# model.export_model('output/conv2d_model.json','output/conv2d_weights.h5',"output/conv2d_emotion_map.json", emotion_map)
