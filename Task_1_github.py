def run_task1():
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models
    import random
    from tensorflow.keras.callbacks import EarlyStopping
    import seaborn as sns

    random.seed(3)
    np.random.seed(0)
    tf.random.set_seed(0)

    dataset_path = r'./Datasets/PneumoniaMNIST/pneumoniamnist.npz'
    data = np.load(dataset_path)

    # Extract training images and labels
    train_images = data['train_images'].reshape(-1, 28, 28, 1)
    train_labels = data['train_labels']
    val_images = data['val_images'].reshape(-1, 28, 28, 1)
    val_labels = data['val_labels']
    test_images = data['test_images'].reshape(-1, 28, 28, 1)
    test_labels = data['test_labels']

    #plotting images
    plt.figure(figsize=(10, 2))
    for i in range(10):  # Range for images
        plt.subplot(1, 10, i + 1)
        plt.imshow(train_images[i], cmap='gray')  # Assuming the images are in grayscale
        plt.title(f'Label: {train_labels[i]}')
        plt.axis('off')
    plt.show()

    # Normalize pixel values to be between 0 and 1
    train_images, val_images, test_images = train_images / 255.0, val_images / 255.0, test_images / 255.0

    # Data augmentation
    def adjust_contrast(images, contrast_factor): # Change contrast
        return tf.map_fn(lambda img: tf.image.adjust_contrast(img, contrast_factor), images)

    contrast_factors = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  # Range of contrast factors used in the plot
    test_accuracies_contrast = []

    contrast = 1.1 # We set this as our main contrast  
    train_images_main = adjust_contrast(train_images, contrast)
    val_images_main = adjust_contrast(val_images, contrast)
    test_images_main = adjust_contrast(test_images, contrast)

    # Defining early stopping 
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, restore_best_weights=True)

    # Defining the model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))  # For binary classification

    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam_optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])

    history = model.fit(train_images_main, train_labels, epochs=25, 
                        batch_size = 32, validation_data=(val_images_main, val_labels),
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
    plt.savefig('accuracy_loss_vs_epoch_early_stopping.png', format='png', dpi=300)
    plt.show()

    test_loss, test_accuracy = model.evaluate(test_images_main,  test_labels, verbose=2)
    print(f"Test accuracy: {test_accuracy}, Test loss: {test_loss}")


    ## plotting test accuracy against contrast factor ##
    for factor in contrast_factors:
        # Adjust contrast
        adjusted_train_images = adjust_contrast(train_images, factor)
        adjusted_val_images = adjust_contrast(val_images, factor)
        adjusted_test_images = adjust_contrast(test_images, factor)

        # Reset the model
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu')) 
        model.add(layers.Dense(1, activation='sigmoid'))  # For binary classification

        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=adam_optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        # Train the model
        model.fit(adjusted_train_images, train_labels, epochs=15, 
                            batch_size=32, validation_data=(adjusted_val_images, val_labels), 
                            verbose=0, callbacks=[early_stopping])  
        
        # Evaluate on the test set
        test_loss, test_accuracy = model.evaluate(adjusted_test_images, test_labels, verbose=0)
        test_accuracies_contrast.append(test_accuracy)
        print(f"Test accuracy: {test_accuracy}, Test loss: {test_loss}")

    # Plot the contrast factors against test accuracies 
    plt.plot(contrast_factors, test_accuracies_contrast, marker='o')
    plt.xlabel('Contrast Factor')
    plt.ylabel('Test Accuracy')
    plt.title('Model Accuracy vs. Contrast Factor on Test data')
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.savefig('accuracy_vs_contrast_factor.png', format='png', dpi=300) 
    plt.show()

    #Plotting performance metrics against different seeds

    def run_model(seed):
        np.random.seed(seed)
        tf.random.set_seed(seed)

        dataset_path =r'./Datasets/PneumoniaMNIST/pneumoniamnist.npz'
        data = np.load(dataset_path)
        # Extract training images and labels
        train_images = data['train_images'].reshape(-1, 28, 28, 1)
        train_labels = data['train_labels']
        val_images = data['val_images'].reshape(-1, 28, 28, 1)
        val_labels = data['val_labels']
        test_images = data['test_images'].reshape(-1, 28, 28, 1)
        test_labels = data['test_labels']

        
        # Normalize pixel values to be between 0 and 1
        train_images, val_images, test_images = train_images / 255.0, val_images / 255.0, test_images / 255.0
        
        # Data augmentation
        def adjust_contrast(images, contrast_factor): # Change contrast
            return tf.map_fn(lambda img: tf.image.adjust_contrast(img, contrast_factor), images)

        contrast = 1.2 # We set this as our main contrast  
        train_images_main = adjust_contrast(train_images, contrast)
        val_images_main = adjust_contrast(val_images, contrast)
        test_images_main = adjust_contrast(test_images, contrast)
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, restore_best_weights=True)

        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))  # For binary classification


        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=adam_optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        model.fit(train_images_main, train_labels, epochs=25, 
                            batch_size = 32, validation_data=(val_images_main, val_labels),
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
    plt.savefig('test_metrics_vs_seeds_Task1.png', dpi=300)
    plt.show()
