class Trainer:
    def __init__(self, criterion, optimizer, scheduler, logger, writer):
        pass

    def train_loop(self, train_loader, val_loader, test_loader, model):
        pass

    def _training_step(self):
        """Traing step for one epoch"""
        pass

    def _validate(self):
        pass

    def _epoch_stats_logging(self):
        pass

    def _intermediate_stats_logging(self):
        pass
