import torch
from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model

class SpectralDecoupling(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps):
        model = initialize_model(config, d_out).to(config.device)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )

    def objective(self, results):
        y, y_hat = results['y_true'], results['y_pred']
        per_sample_losses = torch.log(1.0 + torch.exp(-y_hat[:, 0] * (2.0 * y - 1.0)))
        actual_loss = per_sample_losses.mean()

        lambda_1, lambda_2 = 0.2, 0.2
        gamma_1, gamma_2 = 2.5, 0.44

        actual_loss += lambda_1 / 2 * ((y_hat[torch.where(y == 1)] - gamma_1) ** 2).mean()
        actual_loss += lambda_2 / 2 * ((y_hat[torch.where(y == 0)] - gamma_2) ** 2).mean()

        return actual_loss
