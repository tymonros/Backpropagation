import numpy as np
from random import shuffle
from keras.datasets import mnist
from matplotlib import pyplot
from scipy.stats import norm
import copy
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk, ImageOps


def init_network(n_inputs, n_neurons_per_hidden_layer, num_hidden_layers, n_outputs):
    """
    Initializes the Multi-Layer Perceptron (MLP) network architecture.

    This function sets up the weight matrices for each layer, including bias terms.
    Weights are initialized using the Glorot (Xavier) uniform distribution,
    which helps stabilize gradients during the initial phases of training.

    Args:
        n_inputs (int): Number of input features.
        n_neurons_per_hidden_layer (int): Number of neurons in each hidden layer.
        num_hidden_layers (int): Total number of hidden layers.
        n_outputs (int): Number of output classes.

    Returns:
        list[numpy.ndarray]: A list of NumPy arrays, where each array represents
                             the weights for a specific layer, including the bias column.
    """

    network = []
    inputs_to_current_layer = n_inputs
    for i in range(num_hidden_layers):
        limit = np.sqrt(6.0 / (inputs_to_current_layer + n_neurons_per_hidden_layer))
        hidden_layer_weights = np.random.uniform(
            -limit,
            limit,
            size=(n_neurons_per_hidden_layer, inputs_to_current_layer + 1),
        )
        network.append(hidden_layer_weights)
        inputs_to_current_layer = n_neurons_per_hidden_layer

    limit = np.sqrt(6.0 / (inputs_to_current_layer + n_outputs))
    output_layer_weights = np.random.uniform(
        -limit, limit, size=(n_outputs, inputs_to_current_layer + 1)
    )
    network.append(output_layer_weights)
    return network


def gelu(x):
    """
    Applies the GELU (Gaussian Error Linear Unit) activation function.

    GELU is a smooth non-linear activation function used in hidden layers
    to introduce non-linearity, enabling the network to learn complex relationships.

    Args:
        x (numpy.ndarray): Input values to the activation function.

    Returns:
        numpy.ndarray: Output after applying the GELU activation.
    """
    return x * norm.cdf(x)


def gelu_derivative(x):
    """
    Computes the derivative of the GELU activation function.

    This derivative is essential for the backpropagation algorithm to calculate
    gradients for weights in hidden layers. This implementation uses a common approximation.

    Args:
        x (numpy.ndarray): Input values (pre-activation, z) to the GELU function.

    Returns:
        numpy.ndarray: The derivative of GELU with respect to x.
    """
    return norm.cdf(x) + x * norm.pdf(x)


def softmax(x):
    """
    Applies the Softmax activation function.

    Softmax converts a vector of arbitrary real values into a probability distribution
    over predicted output classes. It ensures numerical stability by
    subtracting the maximum input value before exponentiation.

    Args:
        x (numpy.ndarray): Input array of real values (logits) for the output layer.

    Returns:
        numpy.ndarray: An array of probabilities summing to 1.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def softmax_derivative(softmax_output, expected_output):
    """
    Computes the error term (delta) for the output layer when combined
    with the Cross-Entropy Loss function.

    For Softmax with Cross-Entropy Loss, the derivative simplifies to
    the difference between the predicted probabilities and the true (one-hot encoded) probabilities.

    Args:
        softmax_output (numpy.ndarray): The actual output probabilities from the softmax layer.
        expected_output (numpy.ndarray): The one-hot encoded true label.

    Returns:
        numpy.ndarray: The error term (delta) for the output layer, used in backpropagation.
    """
    return softmax_output - expected_output


def forward_propagate(network, input_row):
    """
    Performs the forward pass through the neural network.

    It computes the activations for each layer sequentially,
    applying weights and activation functions. Intermediate activations
    and pre-activations (weighted sums) are stored for use in backpropagation.

    Args:
        network (list[numpy.ndarray]): The list of weight matrices defining the network.
        input_row (numpy.ndarray): A single input sample.

    Returns:
        tuple[numpy.ndarray, list, list]:
            - final_output (numpy.ndarray): The probabilities from the output layer.
            - all_activations (list[numpy.ndarray]): Activations (outputs) of all layers, including input.
            - all_pre_activations (list[numpy.ndarray]): Pre-activation values (z) for all layers.
    """

    current_input = np.array(input_row).flatten()
    all_activations = [current_input]
    all_pre_activations = []

    for i, layer_weights in enumerate(network):
        biased_input = np.insert(current_input, len(current_input), 1.0)
        z = np.dot(layer_weights, biased_input)
        all_pre_activations.append(z)

        if i == len(network) - 1:
            current_input = softmax(z)
        else:
            current_input = gelu(z)
        all_activations.append(current_input)

    final_output = current_input
    return final_output, all_activations, all_pre_activations


def backward_propagate(network, expected_output, all_activations, all_pre_activations):
    """
    Performs the backward pass (backpropagation) to compute gradients.

    This algorithm calculates the error contribution of each weight in the network,
    propagating the error signal from the output layer back through hidden layers.
    It produces gradients used to update the network's weights.

    Args:
        network (list[numpy.ndarray]): The network's weight matrices.
        expected_output (numpy.ndarray): The one-hot encoded true label for the sample.
        all_activations (list[numpy.ndarray]): List of activations from the forward pass.
        all_pre_activations (list[numpy.ndarray]): List of pre-activation values (z) from the forward pass.

    Returns:
        list[numpy.ndarray]: A list of NumPy arrays, where each array is the gradient
                             matrix for a corresponding layer's weights.
    """
    num_layers = len(network)
    deltas = [None] * num_layers
    gradients = [None] * num_layers

    output_layer_index = num_layers - 1

    deltas[output_layer_index] = softmax_derivative(
        all_activations[output_layer_index + 1], expected_output
    )

    activation_layer_with_bias = np.insert(
        all_activations[output_layer_index],
        len(all_activations[output_layer_index]),
        1.0,
    )
    gradients[output_layer_index] = np.outer(
        deltas[output_layer_index], activation_layer_with_bias
    )

    for i in range(output_layer_index - 1, -1, -1):
        weights_from_current_to_next_layer = network[i + 1][:, :-1]
        error_hidden = np.dot(weights_from_current_to_next_layer.T, deltas[i + 1])
        deltas[i] = error_hidden * gelu_derivative(all_pre_activations[i])

        activation_layer_with_bias = np.insert(
            all_activations[i], len(all_activations[i]), 1.0
        )
        gradients[i] = np.outer(deltas[i], activation_layer_with_bias)

    return gradients


def update_weights(network, gradients, learning_rate):
    """
    Adjusts the network's weights using gradient descent.

    Each weight is updated in the direction that minimizes the loss, scaled by
    the learning rate to control the step size.

    Args:
        network (list[numpy.ndarray]): The network's weight matrices to be updated.
        gradients (list[numpy.ndarray]): The calculated gradients for each weight.
        learning_rate (float): The step size for weight adjustments.
    """
    for i in range(len(network)):
        network[i] -= learning_rate * gradients[i]


def cross_entropy_loss(predictions, targets):
    """
    Computes the categorical cross-entropy loss.

    This loss function quantifies the difference between the predicted probability
    distribution and the true probability distribution (one-hot encoded targets),
    commonly used in multi-class classification. Predictions are clipped to prevent
    logarithm of zero.

    Args:
        predictions (numpy.ndarray): Predicted probabilities from the network's output.
        targets (numpy.ndarray): True labels in one-hot encoded format.

    Returns:
        float: The calculated cross-entropy loss value.
    """
    predictions = np.clip(predictions, 1e-12, 1 - 1e-12)
    loss = -np.sum(targets * np.log(predictions))
    return loss


def predict(network, input_row):
    """
    Generates a prediction for a single input sample using the trained network.

    Performs a forward pass and determines the class with the highest predicted probability.

    Args:
        network (list[numpy.ndarray]): The trained neural network's weight matrices.
        input_row (numpy.ndarray): A single input sample (e.g., a flattened image).

    Returns:
        tuple[numpy.int64, numpy.ndarray]:
            - predicted_digit (numpy.int64): The index of the predicted class (0-9).
            - probabilities (numpy.ndarray): The full array of probabilities for all output classes.
    """
    final_output, _, _ = forward_propagate(network, input_row)
    return np.argmax(final_output), final_output


def calculate_accuracy(network, X_data, y_data):
    """
    Evaluates the accuracy of the neural network on a given dataset.

    It calculates the percentage of correctly classified samples
    by comparing the network's predictions against the true labels.

    Args:
        network (list[numpy.ndarray]): The trained neural network.
        X_data (numpy.ndarray): Input features (e.g., images) to evaluate.
        y_data (numpy.ndarray): True labels corresponding to the input features.

    Returns:
        float: The accuracy as a percentage (0-100%).
    """
    correct_predictions = 0
    num_samples_to_evaluate = len(X_data)
    for i in range(num_samples_to_evaluate):
        prediction, _ = predict(network, X_data[i])
        if prediction == y_data[i]:
            correct_predictions += 1
    return (correct_predictions / num_samples_to_evaluate) * 100


def train_network(
    network, train_X, train_y, test_X, test_y, learning_rate, n_epochs, evaluation_frequency=1000
):
    """
    Trains the neural network using stochastic gradient descent.

    The network processes the training data for a specified number of epochs.
    In each iteration, a sample is forward-propagated, its error is computed,
    gradients are backpropagated, and weights are updated. Training accuracy
    is periodically evaluated and plotted.

    Args:
        network (list[numpy.ndarray]): The neural network model to be trained.
        train_X (numpy.ndarray): Training input features.
        train_y (numpy.ndarray): True labels for the training data.
        learning_rate (float): The step size for weight updates.
        n_epochs (int): The number of complete passes over the training dataset.
        evaluation_frequency (int, optional): Frequency (in samples processed)
                                              for evaluating and printing training accuracy. Defaults to 1000.

    Returns:
        list[numpy.ndarray]: The trained neural network's weight matrices.
    """
    test_len = len(test_X)
    num_samples = len(train_X)
    accuracies = []
    sample_indices_for_plot = []

    for epoch in range(n_epochs):
        print(f"Starting Epoch {epoch + 1}/{n_epochs}")
        indices = list(range(num_samples))
        shuffle(indices)

        for i, idx in enumerate(indices):
            input_data = train_X[idx]
            label = train_y[idx]

            expected_output = np.zeros(network[-1].shape[0])
            expected_output[label] = 1

            final_output, all_activations, all_pre_activations = forward_propagate(
                network, input_data
            )
            gradients = backward_propagate(
                network, expected_output, all_activations, all_pre_activations
            )
            update_weights(network, gradients, learning_rate)

            if (i + 1) % evaluation_frequency == 0:
                subset_size = min(evaluation_frequency * 5, test_len)
                if subset_size > 0:
                    random_subset_indices = np.random.choice(
                        test_len, subset_size, replace=False
                    )
                    current_accuracy = calculate_accuracy(
                        network,
                        test_X[random_subset_indices],
                        test_y[random_subset_indices],
                    )
                else:
                    current_accuracy = 0.0

                accuracies.append(current_accuracy)
                sample_indices_for_plot.append(epoch * num_samples + i)
                print(
                    f"  Processed {i + 1}/{num_samples} samples in epoch {epoch + 1}. Current Training Accuracy (random subset): {current_accuracy:.2f}%"
                )

        epoch_accuracy = calculate_accuracy(network, train_X, train_y)
        print(
            f"Epoch {epoch + 1} completed. Full Training Accuracy: {epoch_accuracy:.2f}%"
        )

    pyplot.figure(figsize=(10, 6))
    pyplot.plot(sample_indices_for_plot, accuracies)
    pyplot.title("Training Accuracy Over Samples (Subset Evaluation)")
    pyplot.xlabel("Total Samples Processed")
    pyplot.ylabel("Accuracy (%)")
    pyplot.grid(True)
    pyplot.show()

    return network


class DigitRecognizerApp:
    """
    A Tkinter-based graphical user interface (GUI) for interactive digit recognition.

    Users can draw digits on a canvas, and the application will preprocess the drawing
    and use a trained neural network to predict the digit. It also allows loading
    and analyzing pre-existing MNIST examples.
    """

    def __init__(self, master, network, test_X, test_y):
        """
        Initializes the DigitRecognizerApp GUI components and binds event handlers.

        Args:
            master (tk.Tk): The root Tkinter window.
            network (list[numpy.ndarray]): The trained neural network model.
            test_X (numpy.ndarray): MNIST test images for loading examples.
            test_y (numpy.ndarray): True labels for MNIST test images.
        """
        self.master = master
        self.network = network
        self.test_X = test_X
        self.test_y = test_y
        self.master.title("Draw Digit for Recognition")

        self.canvas_width = 280
        self.canvas_height = 280
        self.pen_thickness = 15

        self.canvas = tk.Canvas(
            master,
            bg="black",
            width=self.canvas_width,
            height=self.canvas_height,
            bd=0,
            highlightthickness=0,
        )
        self.canvas.pack(pady=10)

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)

        self.last_x, self.last_y = None, None

        button_frame = tk.Frame(master)
        button_frame.pack(pady=5)

        self.predict_button = tk.Button(
            button_frame, text="Analyze Number", command=self.analyze_number
        )
        self.predict_button.pack(side=tk.LEFT, padx=5)

        self.load_mnist_button = tk.Button(
            button_frame,
            text="Load MNIST Example",
            command=self.load_random_mnist_example,
        )
        self.load_mnist_button.pack(side=tk.LEFT, padx=5)

        self.clear_button = tk.Button(
            button_frame, text="Clear", command=self.clear_canvas
        )
        self.clear_button.pack(side=tk.RIGHT, padx=5)

        self.result_label = tk.Label(
            master, text="Predicted: -\nProbabilities:\n", font=("Helvetica", 12)
        )
        self.result_label.pack(pady=10)

        self.processed_image_label = tk.Label(master)
        self.processed_image_label.pack(pady=5)

    def start_draw(self, event):
        """
        Initiates a drawing stroke by recording the initial mouse coordinates.
        """
        self.last_x, self.last_y = event.x, event.y
        self.draw_line(event)

    def draw_line(self, event):
        """
        Draws a line segment on both the Tkinter canvas and the internal Pillow image.

        This method is called continuously as the mouse is dragged with the left button pressed.
        """
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y,
                fill="white",
                width=self.pen_thickness,
                capstyle=tk.ROUND,
                smooth=tk.TRUE,
            )
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=255,
                width=self.pen_thickness,
                joint="round",
            )
            self.last_x, self.last_y = event.x, event.y

    def stop_draw(self, event):
        """
        Terminates the current drawing stroke by resetting the last recorded mouse coordinates.
        """
        self.last_x, self.last_y = None, None

    def clear_canvas(self):
        """
        Clears all content from the drawing canvas and resets associated display elements.
        """
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Predicted: -\nProbabilities:\n")
        self.processed_image_label.config(image=None)

    def analyze_number(self):
        """
        Processes the drawn image, applies necessary transformations to match
        MNIST dataset characteristics, performs inference using the neural network,
        and displays the prediction results.
        """
        bbox = self.image.getbbox()

        if bbox:  # If something was drawn
            cropped_image = self.image.crop(bbox)

            width, height = cropped_image.size
            if width > height:
                new_width = 20
                new_height = int(height * (new_width / width))
            else:
                new_height = 20
                new_width = int(width * (new_height / height))

            temp_resized_image = cropped_image.resize(
                (new_width, new_height), Image.LANCZOS
            )

            final_28x28_image = Image.new("L", (28, 28), 0)
            paste_x = (28 - new_width) // 2
            paste_y = (28 - new_height) // 2
            final_28x28_image.paste(temp_resized_image, (paste_x, paste_y))

        else:
            final_28x28_image = Image.new("L", (28, 28), 0)

        processed_array = np.array(final_28x28_image, dtype=np.float32) / 255.0

        display_img = Image.fromarray((processed_array * 255).astype(np.uint8))
        display_img_tk = ImageTk.PhotoImage(
            display_img.resize((100, 100), Image.NEAREST)
        )
        self.processed_image_label.config(image=display_img_tk)
        self.processed_image_label.image = display_img_tk

        input_for_prediction = processed_array.flatten()

        predicted_digit_np, probabilities = predict(self.network, input_for_prediction)

        prediction_display = int(predicted_digit_np)

        probabilities_text = "Probabilities:\n"
        for i, prob in enumerate(probabilities):
            probabilities_text += f"{i}: {prob * 100:.2f}%  "
            if (i + 1) % 5 == 0:
                probabilities_text += "\n"

        self.result_label.config(
            text=f"Predicted: {prediction_display}\n{probabilities_text}"
        )

    def load_random_mnist_example(self):
        """
        Loads and displays a random MNIST test set example on the canvas.

        This method facilitates quick testing of the model's performance on
        standardized dataset images and updates the display accordingly.
        """
        self.clear_canvas()

        random_idx = np.random.randint(0, len(self.test_X))
        mnist_image_array = self.test_X[random_idx]
        true_label = self.test_y[random_idx]

        mnist_image_pil = Image.fromarray(
            (mnist_image_array * 255).astype(np.uint8), mode="L"
        )

        # Display the 28x28 MNIST image directly on the canvas, scaled up
        display_on_canvas_pil = mnist_image_pil.resize(
            (self.canvas_width, self.canvas_height), Image.NEAREST
        )

        # Also copy this image to our internal self.image for analysis consistency
        self.image = display_on_canvas_pil.copy()
        self.draw = ImageDraw.Draw(
            self.image
        )  # Reinitialize ImageDraw for the new image

        self.tk_img = ImageTk.PhotoImage(display_on_canvas_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        self.canvas.image = self.tk_img

        self.result_label.config(text=f"Loaded MNIST: {true_label} (Click Analyze)")
        # Now automatically analyze to show immediate result
        self.analyze_number()


# --- Main Execution Block (no changes needed) ---
if __name__ == "__main__":
    """
    Main entry point for the application.

    Initializes the neural network, loads and preprocesses the MNIST dataset,
    trains the network, evaluates its performance, and launches the
    interactive GUI for digit recognition.
    """
    # ... (rest of your main execution block) ...
    # It starts here with init_network and ends with root.mainloop()
    network = init_network(784, 28, 2, 10)

    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    train_X = train_X.astype("float32") / 255.0
    test_X = test_X.astype("float32") / 255.0

    print("X_train: " + str(train_X.shape))
    print("Y_train: " + str(train_y.shape))
    print("X_test:  " + str(test_X.shape))
    print("Y_test:  " + str(test_y.shape))

    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(train_X[i], cmap=pyplot.get_cmap("gray"))
    pyplot.show()

    print("\nProceeding with training...")
    network = train_network(
        network, train_X, train_y, test_X, test_y, 0.005, 3, evaluation_frequency=1000
    )

    print("\n--- Final Evaluation on Test Set ---")
    test_accuracy = calculate_accuracy(network, test_X, test_y)
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")

    print("\nLaunching drawing application...")
    root = tk.Tk()
    app = DigitRecognizerApp(root, network, test_X, test_y)
    root.mainloop()
