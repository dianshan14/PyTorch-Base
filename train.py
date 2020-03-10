from tensorboardX import SummaryWriter

from utils.optim import get_lr_scheduler, get_optimizer
from utils.util import get_logger
from utils.train_functions import Trainer

def train():
    logger = get_logger("./logger")
    writer = SummaryWriter("./temp.tb")


    train_loader, val_loader = None, None
    test_loader = None

    model = None

    criterion = None
    optimizer = get_optimizer(model)
    scheduler = get_lr_scheduler(optimizer)

    trainer = Trainer(criterion, optimizer, scheduler, logger, writer)
    trainer.train_loop(train_loader, val_loader, test_loader, model)

if __name__ == "__main__":
    train()
