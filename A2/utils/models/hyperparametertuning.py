import optuna
import torch
from utils.models.CNN import CNNModel
from utils.models.loss import NFLandRCE
from utils.dataloader.FashionMNISTDataLoader import FashionMNISTDataLoader, CIFAR10DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class OptunaOptimization:
    def __init__(self, path, 
                 device, t_matrix, 
                 repetitions = 3,
                 sample_size=0.005, 
                 study_name="CNN_Hyperparameter_Tuning", 
                 weights_path="weights",    
                 sample_shape = (1, 28, 28)):
        """
        Initializes the OptunaOptimization class.

        Args:
            path (str): Path to the dataset file.
            device (torch.device): Computing device ('mps' or 'cpu').
            t_matrix (numpy.ndarray or torch.Tensor): Transition matrix for the model.
            sample_size (float, optional): Fraction of data to use for training. Default is 0.005 (0.5%).
        """
        self.path = path
        self.device = device
        self.t_matrix = t_matrix
        self.sample_size = sample_size
        self.study_name = study_name
        self.repetitions = repetitions
        self.sample_shape = sample_shape
        self.weights_path = weights_path

    def objective(self, trial):
        """
        Objective function for Optuna to optimize.

        Args:
            trial (optuna.trial.Trial): Optuna trial object.

        Returns:
            float: Validation accuracy.
        """
        # 1. Define the hyperparameters to be tuned
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
        kernel_size_conv1 = trial.suggest_categorical('kernel_size_conv1', [2, 3, 4, 5])
        conv1_channels = trial.suggest_categorical('conv1_channels', [32, 64, 128])
        conv2_channels = trial.suggest_categorical('conv2_channels', [64, 128, 256])
        fc_size = trial.suggest_categorical('fc_size', [64, 128, 256])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        epochs = trial.suggest_int('epochs', 20, 50)
        add_l1 = trial.suggest_categorical('add_l1', [True, False])
        criterion = trial.suggest_categorical('criterion', ['cross_entropy', 'nf_land_rce'])

        # Optional hyperparameter: L1 regularization lambda
        if add_l1:
            lambda_l1 = trial.suggest_loguniform('lambda_l1', 1e-5, 1e-2)
        else:
            lambda_l1 = 0.0

        # 3. Initialize the model with sampled hyperparameters
        model = CNNModel(
            t_matrix=self.t_matrix,
            device=self.device,
            dropout=dropout,
            kernel_size_pool=2,  # Fixed pooling size
            kernel_size_conv=kernel_size_conv1,  # First conv layer kernel size
            conv_channels=[conv1_channels, conv2_channels],  # 2 conv layers
            use_batch_norm=use_batch_norm,
            l1_lambda=lambda_l1,
            fc_layers_sizes=[fc_size],  # 1 FC layer
            num_classes=4,  # Adjust based on your dataset labels
            input_shape = self.sample_shape  # Adjust based on your dataset shape
            
        ).to(self.device)



        accuracy_list = []

        for _ in range(self.repetitions):
            # 2. Initialize DataLoaders with the sampled batch size and sample size
            data_loader = FashionMNISTDataLoader(
                path=self.path,
                batch_size=batch_size,
                sample_size=self.sample_size,
                train_percentage=0.8,
                device=self.device
            ) if 'Fashion' in self.path else CIFAR10DataLoader(
                path=self.path,
                batch_size=batch_size,
                sample_size=self.sample_size,
                train_percentage=0.8,
                device=self.device
            )
            
            train_loader, eval_loader, test_loader = data_loader.get_loaders()
            # 4. Define optimizer
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # 5. Define loss criterion
            if criterion == 'cross_entropy':
                criterion = nn.CrossEntropyLoss()
            elif criterion == 'nf_land_rce':
                criterion = NFLandRCE(alpha=1, beta=1, num_classes=model.num_classes)

            # add mae   
            
            # 6. Training loop
            for epoch in range(epochs):
                model.train()  # Set model to training mode
                model.training = True
                running_loss = 0.0
                for images, labels in train_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # Add L1 regularization if specified
                    if add_l1:
                        l1_loss = 0.0
                        for param in model.parameters():
                            l1_loss += param.abs().sum()
                        loss += lambda_l1 * l1_loss

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                # 7. Validation after each epoch
                model.eval()  # Set model to evaluation mode
                model.training = True
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in eval_loader:
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                accuracy = correct / total

                # 8. Report intermediate objective value to Optuna
                accuracy_list.append(accuracy)
               

        mean_accuracy = np.mean(np.array(accuracy_list))
        std_accuracy = np.std(np.array(accuracy_list))

        trial.set_user_attr('mean_accuracy', mean_accuracy) 
        trial.set_user_attr('std_accuracy', std_accuracy)

        # store the msave_modelodel.pth with the best accuracy
        torch.save(model.state_dict(), f"{self.weights_path}/model{mean_accuracy}.pth")

        return mean_accuracy
    

    def run_optimization(self, n_trials=50):
        """
        Runs the Optuna optimization process.

        Args:
            n_trials (int, optional): Number of trials for optimization. Default is 50.

        Returns:
            optuna.trial.FrozenTrial: The best trial found.
        """
        # 1. Create an Optuna study
        study = optuna.create_study(
            direction="maximize",  # We aim to maximize validation accuracy
            study_name=self.study_name,
            storage="sqlite:///cnn_hyperparameter_tuning.db",
            load_if_exists=True
        )

        # 2. Start optimization
        study.optimize(self.objective, n_trials=n_trials, timeout=None)

        # 3. Output the best trial
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print(f"  Validation Accuracy: {trial.value:.4f}")
        print("  Best hyperparameters: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return trial