def run_task2():    
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import datasets, layers, models
    import random
    from tensorflow.keras.callbacks import EarlyStopping
    import seaborn as sns

    random.seed(0)
    np.random.seed(0)
    tf.random.set_seed(0)

    dataset_path = r'./Datasets/PathMNIST/pathmnist.npz'
    data = np.load(dataset_path)

    # Extract training images and labels
    train_images = data['train_images']
    train_labels = data['train_labels']
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    # Normalize pixel values to be between 0 and 1
    train_images, val_images, test_images = train_images / 255.0, val_images / 255.0, test_images / 255.0

    # Data augmentation
    def saturation(images, saturation_factor): #image saturation
        return tf.map_fn(lambda img: tf.image.adjust_saturation(img, saturation_factor), images)

    saturation_factor = 2
    train_images_main = saturation(train_images, saturation_factor)
    val_images_main = saturation(val_images, saturation_factor)
    test_images_main = saturation(test_images, saturation_factor)

    def adjust_contrast(images, contrast_factor): # Change contrast
        return tf.map_fn(lambda img: tf.image.adjust_contrast(img, contrast_factor), images)

    contrast = 1.2 # We set this as our main contrast  
    train_images_main = adjust_contrast(train_images_main, contrast)
    val_images_main = adjust_contrast(val_images_main, contrast)
    test_images_main = adjust_contrast(test_images_main, contrast)

    # Defining early stopping 
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, restore_best_weights=True)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(128, (2, 2), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu')) 
    model.add(layers.Dense(9, activation='softmax')) # softmax turns the outputs into probabilities that sum up to 1, making it suitable for multi-class classification.

    model.summary()

    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0007)
    model.compile(optimizer=adam_optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    history = model.fit(train_images_main, train_labels, epochs=25, 
                        validation_data=(val_images_main, val_labels),
                        callbacks=[early_stopping])


    # plotting accuracy 
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Validation accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.legend(loc='lower right')

    # plottinig Loss
    plt.subplot(1, 2, 2)  
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 1])
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.legend(loc='upper right')

    plt.tight_layout() 
    plt.savefig('accuracy_loss_vs_epoch_early_stopping_task2.png', format='png', dpi=300)
    plt.show()
    test_loss, test_accuracy = model.evaluate(test_images_main,  test_labels, verbose=2)
    print(f"Test accuracy: {test_accuracy}, Test loss: {test_loss}")


    ## plotting for different seeds ##
    def run_model(seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)

        dataset_path = r'./Datasets/PathMNIST/pathmnist.npz'
        data = np.load(dataset_path)
        # Extract training images and labels
        train_images = data['train_images']
        train_labels = data['train_labels']
        val_images = data['val_images']
        val_labels = data['val_labels']
        test_images = data['test_images']
        test_labels = data['test_labels']
        
        # Data augmentation
        def saturation(images, saturation_factor): #image saturation
            return tf.map_fn(lambda img: tf.image.adjust_saturation(img, saturation_factor), images)

        saturation_factor = 2
        train_images_main = saturation(train_images, saturation_factor)
        val_images_main = saturation(val_images, saturation_factor)
        test_images_main = saturation(test_images, saturation_factor)

        def adjust_contrast(images, contrast_factor): # Change contrast
            return tf.map_fn(lambda img: tf.image.adjust_contrast(img, contrast_factor), images)

        contrast = 1.2 # We set this as our main contrast  
        train_images_main = adjust_contrast(train_images_main, contrast)
        val_images_main = adjust_contrast(val_images_main, contrast)
        test_images_main = adjust_contrast(test_images_main, contrast)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, restore_best_weights=True)

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.Conv2D(128, (2, 2), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'), 
            layers.Dense(9, activation='softmax') # For multi-class classification
        ])

        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0007)
        model.compile(optimizer=adam_optimizer,
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        model.fit(train_images_main, train_labels, epochs=25, 
                        validation_data=(val_images_main, val_labels),
                        callbacks=[early_stopping])

        test_loss, test_accuracy = model.evaluate(test_images_main, test_labels, verbose=2)
        return test_loss, test_accuracy

    seeds = [0, 1, 2, 3, 4]  # List of seeds
    all_test_accuracies = []
    all_test_losses = []

    for seed in seeds:
        test_loss, test_accuracy = run_model(seed)
        all_test_accuracies.append(test_accuracy)
        all_test_losses.append(test_loss)

    # Calculate averages
    avg_test_accuracy = np.mean(all_test_accuracies)
    avg_test_loss = np.mean(all_test_losses)

    print(f"Average Test Accuracy: {avg_test_accuracy}, Average Test Loss: {avg_test_loss}")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(seeds, all_test_accuracies, marker='o', color='b')
    plt.title('Test Accuracy for Different Seeds')
    plt.xlabel('Seed')
    plt.ylabel('Test Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(seeds, all_test_losses, marker='o', color='r')
    plt.title('Test Loss for Different Seeds')
    plt.xlabel('Seed')
    plt.ylabel('Test Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('test_metrics_vs_seeds.png', dpi=300)
    plt.show()


    ## plotting the heatmaps ##
    # Extract training images and labels
    train_images = data['train_images']
    train_labels = data['train_labels']
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    # Normalize pixel values to be between 0 and 1
    train_images, val_images, test_images = train_images / 255.0, val_images / 255.0, test_images / 255.0

    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, restore_best_weights=True)


    def adjust_saturation_and_contrast(images, saturation_factor, contrast_factor):
        images = tf.image.adjust_saturation(images, saturation_factor)
        images = tf.image.adjust_contrast(images, contrast_factor)
        return images

    saturation_factors = [1, 2, 3, 4]
    contrast_factors = [0.9, 1.0, 1.1, 1.2, 1.3]
    accuracy_results = np.zeros((len(saturation_factors), len(contrast_factors)))
    loss_results = np.zeros((len(saturation_factors), len(contrast_factors)))

    for i, sat in enumerate(saturation_factors):
        for j, con in enumerate(contrast_factors):
            print(f"Evaluating with saturation {sat} and contrast {con}")

            # Adjust train, validation, and test images
            adj_train = adjust_saturation_and_contrast(train_images, sat, con)
            adj_val = adjust_saturation_and_contrast(val_images, sat, con)
            adj_test = adjust_saturation_and_contrast(test_images, sat, con)

            model = models.Sequential()
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            model.add(layers.Conv2D(128, (2, 2), activation='relu'))

            model.add(layers.Flatten())
            model.add(layers.Dense(128, activation='relu')) 
            model.add(layers.Dense(9, activation='softmax')) # softmax turns the outputs into probabilities that sum up to 1, making it suitable for multi-class classification.

            model.summary()
            
            adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0007)
            model.compile(optimizer=adam_optimizer,
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
            model.fit(adj_train, train_labels, epochs=25, 
                            validation_data=(adj_val, val_labels),
                            callbacks=[early_stopping])

            # Evaluate the model
            loss, accuracy = model.evaluate(adj_test, test_labels, verbose=2)
            accuracy_results[i, j] = accuracy
            loss_results[i, j] = loss

    # Plotting Heatmaps
    plt.figure(figsize=(10, 8))
    sns.heatmap(accuracy_results, annot=True, xticklabels=contrast_factors, yticklabels=saturation_factors, fmt=".2f")
    plt.title("Accuracy Heatmap")
    plt.xlabel("Contrast Factor")
    plt.ylabel("Saturation Factor")
    plt.savefig('accuracy_heatmap.png', dpi=300)
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(loss_results, annot=True, xticklabels=contrast_factors, yticklabels=saturation_factors, fmt=".2f")
    plt.title("Loss Heatmap")
    plt.xlabel("Contrast Factor")
    plt.ylabel("Saturation Factor")
    plt.savefig('loss_heatmap.png', dpi=300)
    plt.show()
