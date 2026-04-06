# Import necessary packages
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary as MS
import torch


class ModelSummaryLogger(pl.Callback):
    def __init__(self, max_depth=2, log_graph=True):
        self.max_depth = max_depth
        self.log_graph = log_graph

    def on_fit_start(self, trainer, pl_module):
        if not (trainer.logger and hasattr(trainer.logger, 'experiment')):
            return

        experiment = trainer.logger.experiment

        # log text summary
        summary = MS(pl_module, max_depth=self.max_depth)
        self.log_text_summary(experiment, summary, pl_module)

        # log model graph if requested
        if self.log_graph:
            self.log_model_graph(experiment, pl_module, trainer)

        # log parameter stats
        self.log_parameter_stats(experiment, pl_module)

        # log model hyperparameters
        self.log_model_hyperparameters(trainer, pl_module)

    def log_text_summary(self, experiment, summary, pl_module):
        # main summary log
        experiment.add_text(
            'Model_Info/Architecture_Summary',
            f'<pre>{str(summary)}</pre>',
            global_step=0
        )

        # model info
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)

        stats_text = f'''
        MODEL STATISTICS:
        ================
        Total Parameters: {total_params:,}
        Trainable Parameters: {trainable_params:,}
        Non-trainable Parameters: {total_params - trainable_params:,}
        Model Size (MB): {total_params * 4 / (1024 ** 2):.2f}
        Memory Usage (MB): {total_params * 4 / (1024 ** 2) * 2:.2f} (approx. with gradients)
        '''

        experiment.add_text(
            'Model_Info/Statistics',
            f'<pre>{stats_text}</pre>',
            global_step=0
        )

    def log_model_graph(self, experiment, pl_module, trainer):
        try:
            # create a sample input
            if hasattr(pl_module, 'example_input_array') and pl_module.example_input_array is not None:
                sample_input = pl_module.example_input_array
            else:
                # try to infer input shape
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                sample_input = torch.randn(1, 1, 256, 256).to(device)

            experiment.add_graph(pl_module, sample_input)
        except Exception as e:
            print(f'Could not log model graph: {e}')

    def log_parameter_stats(self, experiment, pl_module):
        total_params = sum(p.numel() for p in pl_module.parameters())
        trainable_params = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)

        experiment.add_scalar('Model_Stats/Total_Parameters', total_params, 0)
        experiment.add_scalar('Model_Stats/Trainable_Parameters', trainable_params, 0)
        experiment.add_scalar('Model_Stats/Model_Size_MB', total_params * 4 / (1024 ** 2), 0)

    def log_model_hyperparameters(self, trainer, pl_module):
        if hasattr(pl_module, 'hparams') and pl_module.hparams:
            hparams_text = 'MODEL HYPERPARAMETERS:\n' + '=' * 30 + '\n'
            for key, value in pl_module.hparams.items():
                hparams_text += f'{key}: {value}\n'

            trainer.logger.experiment.add_text(
                'Model_Info/Hyperparameters',
                f'<pre>{hparams_text}</pre>',
                global_step=0
            )