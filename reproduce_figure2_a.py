import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd

# Ensure result directory exists
RESULT_DIR = r'd:\HKU\sem2\7404_final\result'
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# Configuration
BATCH_SIZE = 128
EPOCHS = 20 
LR = 0.001
EPSILON = 0.25 
N_NETS = 15 

# Load MNIST
print("Loading MNIST...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# One-hot encode labels
y_train_onehot = tf.keras.utils.to_categorical(y_train, 10)
y_test_onehot = tf.keras.utils.to_categorical(y_test, 10)

def create_mlp_model(dropout_rate=0.0):
    inputs = tf.keras.Input(shape=(784,))
    x = inputs
    # 3 hidden layers of 200 units
    for _ in range(3):
        x = tf.keras.layers.Dense(200, activation='relu')(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

class Trainer:
    def __init__(self, model, method='standard', epsilon=0.25):
        self.model = model
        self.method = method
        self.epsilon = epsilon
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

    @tf.function
    def train_step_standard(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.loss_fn(y, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    @tf.function
    def train_step_adv(self, x, y):
        with tf.GradientTape() as tape_adv:
            tape_adv.watch(x)
            logits = self.model(x, training=True)
            loss = self.loss_fn(y, logits)
        
        grad = tape_adv.gradient(loss, x)
        signed_grad = tf.sign(grad)
        x_adv = x + self.epsilon * signed_grad
        x_adv = tf.clip_by_value(x_adv, 0.0, 1.0)
        
        with tf.GradientTape() as tape:
            logits_clean = self.model(x, training=True)
            loss_clean = self.loss_fn(y, logits_clean)
            
            logits_adv = self.model(x_adv, training=True)
            loss_adv = self.loss_fn(y, logits_adv)
            
            total_loss = 0.5 * loss_clean + 0.5 * loss_adv
            
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss

    @tf.function
    def train_step_random(self, x, y):
        noise = tf.random.uniform(tf.shape(x), minval=-self.epsilon, maxval=self.epsilon)
        x_noise = x + noise
        x_noise = tf.clip_by_value(x_noise, 0.0, 1.0)
        
        with tf.GradientTape() as tape:
            logits_clean = self.model(x, training=True)
            loss_clean = self.loss_fn(y, logits_clean)
            
            logits_noise = self.model(x_noise, training=True)
            loss_noise = self.loss_fn(y, logits_noise)
            
            total_loss = 0.5 * loss_clean + 0.5 * loss_noise
            
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            for x_batch, y_batch in dataset:
                if self.method == 'standard' or self.method == 'mc_dropout':
                    self.train_step_standard(x_batch, y_batch)
                elif self.method == 'adv':
                    self.train_step_adv(x_batch, y_batch)
                elif self.method == 'random':
                    self.train_step_random(x_batch, y_batch)

def calculate_metrics(y_true_onehot, y_pred_probs):
    """
    Calculate Classification Error, NLL, and Brier Score
    y_true_onehot: (N, 10)
    y_pred_probs: (N, 10)
    """
    # 1. Classification Error
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    error = 1.0 - np.mean(y_true == y_pred)
    
    # 2. NLL (Negative Log Likelihood)
    y_pred_probs_clipped = np.clip(y_pred_probs, 1e-15, 1 - 1e-15)
    correct_probs = np.sum(y_true_onehot * y_pred_probs_clipped, axis=1)
    nll = -np.mean(np.log(correct_probs))
    
    # 3. Brier Score
    bs_sum = np.sum((y_pred_probs - y_true_onehot)**2, axis=1) 
    brier_score = np.mean(bs_sum) / 10.0 # Scale by C=10 as per tool research
    
    return error, nll, brier_score

def evaluate_models_metrics(models, x, y_true_onehot, method='ensemble', n_samples=N_NETS):
    preds = []
    
    if method == 'mc_dropout':
        model = models[0]
        print("Sampling MC Dropout...")
        for i in range(n_samples):
            batch_preds = []
            ds = tf.data.Dataset.from_tensor_slices(x).batch(256)
            for batch in ds:
                batch_preds.append(model(batch, training=True).numpy())
            p = np.concatenate(batch_preds, axis=0)
            preds.append(p)
    else:
        for i, model in enumerate(models):
            p = model.predict(x, batch_size=256, verbose=0)
            preds.append(p)
            
    preds = np.array(preds) # (M, N, 10)
    
    errors = []
    nlls = []
    brier_scores = []
    
    print("Calculating metrics...")
    for m in range(1, n_samples + 1):
        # Average predictions of first m models/samples
        avg_preds = np.mean(preds[:m], axis=0) 
        err, nll, bs = calculate_metrics(y_true_onehot, avg_preds)
        errors.append(err)
        nlls.append(nll)
        brier_scores.append(bs)
        
    return errors, nlls, brier_scores

def main():
    # Dataset
    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_onehot))
    ds = ds.shuffle(60000).batch(BATCH_SIZE)
    
    # Initialize metric storage
    results = {}
    methods = ['Ensemble', 'Ensemble + R', 'Ensemble + AT', 'MC dropout']
    for m in methods:
        results[m] = {'error': [], 'nll': [], 'bs': []}

    # 1. Ensemble
    print("\n--- 1. Training Ensemble ---")
    models = []
    for i in range(N_NETS):
        print(f"Net {i+1}/{N_NETS}")
        model = create_mlp_model()
        trainer = Trainer(model, method='standard')
        trainer.train(ds, EPOCHS)
        models.append(model)
    e, n, b = evaluate_models_metrics(models, x_test, y_test_onehot)
    results['Ensemble'] = {'error': e, 'nll': n, 'bs': b}
    
    # 2. Ensemble + R
    print("\n--- 2. Training Ensemble + R ---")
    models = []
    for i in range(N_NETS):
        print(f"Net {i+1}/{N_NETS}")
        model = create_mlp_model()
        trainer = Trainer(model, method='random', epsilon=EPSILON)
        trainer.train(ds, EPOCHS)
        models.append(model)
    e, n, b = evaluate_models_metrics(models, x_test, y_test_onehot)
    results['Ensemble + R'] = {'error': e, 'nll': n, 'bs': b}
    
    # 3. Ensemble + AT
    print("\n--- 3. Training Ensemble + AT ---")
    models = []
    for i in range(N_NETS):
        print(f"Net {i+1}/{N_NETS}")
        model = create_mlp_model()
        trainer = Trainer(model, method='adv', epsilon=EPSILON)
        trainer.train(ds, EPOCHS)
        models.append(model)
    e, n, b = evaluate_models_metrics(models, x_test, y_test_onehot)
    results['Ensemble + AT'] = {'error': e, 'nll': n, 'bs': b}
    
    # 4. MC Dropout
    print("\n--- 4. Training MC Dropout ---")
    model = create_mlp_model(dropout_rate=0.5)
    trainer = Trainer(model, method='mc_dropout')
    trainer.train(ds, EPOCHS * 2)
    e, n, b = evaluate_models_metrics([model], x_test, y_test_onehot, method='mc_dropout', n_samples=N_NETS)
    results['MC dropout'] = {'error': e, 'nll': n, 'bs': b}

    # Save Data
    data = {}
    x_axis = range(1, N_NETS + 1)
    for method in results:
        data[f'{method}_Error'] = results[method]['error']
        data[f'{method}_NLL'] = results[method]['nll']
        data[f'{method}_Brier'] = results[method]['bs']
    
    df = pd.DataFrame(data, index=x_axis)
    df.to_csv(os.path.join(RESULT_DIR, 'figure2_metrics_data.csv'))
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['error', 'nll', 'bs']
    titles = ['Classification Error (%)', 'NLL', 'Brier Score']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        # For error, multiply by 100
        factor = 100 if metric == 'error' else 1
        
        ax.plot(x_axis, np.array(results['Ensemble'][metric]) * factor, 'r-', label='Ensemble')
        ax.plot(x_axis, np.array(results['Ensemble + R'][metric]) * factor, 'gray', label='Ensemble + R')
        ax.plot(x_axis, np.array(results['Ensemble + AT'][metric]) * factor, 'b-', label='Ensemble + AT')
        ax.plot(x_axis, np.array(results['MC dropout'][metric]) * factor, 'g-', label='MC dropout')
        
        ax.set_title(titles[i])
        ax.set_xlabel('Number of nets')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(RESULT_DIR, 'figure2_metrics_repro.png')
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to {save_path}")

if __name__ == '__main__':
    main()
