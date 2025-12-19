from tensorflow.keras.preprocessing.image import ImageDataGenerator
import yaml

def get_augmentation_generator(config_path='config/training_config.yaml'):
    """Get data augmentation generator from config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    aug_config = config['augmentation']
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=aug_config['rotation_range'],
        width_shift_range=aug_config['width_shift_range'],
        height_shift_range=aug_config['height_shift_range'],
        horizontal_flip=aug_config['horizontal_flip'],
        zoom_range=aug_config['zoom_range'],
        fill_mode=aug_config['fill_mode']
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_test_datagen

def create_data_generators(train_dir, val_dir, test_dir, batch_size=32, target_size=(224, 224)):
    """Create data generators for train, validation, and test"""
    train_datagen, val_test_datagen = get_augmentation_generator()
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        color_mode='rgb'
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        color_mode='rgb'
    )
    
    return train_generator, val_generator, test_generator
