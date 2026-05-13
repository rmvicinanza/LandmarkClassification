import torch
import torch.nn as nn
import torch.optim


def get_loss():
    """
    Get an instance of the CrossEntropyLoss (useful for classification),
    optionally moving it to the GPU if use_cuda is set to True
    """

    # YOUR CODE HERE: select a loss appropriate for classification
    loss  = nn.CrossEntropyLoss()

    return loss


def get_optimizer(
    model: nn.Module,
    optimizer: str = "SGD",
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0,
):
    """
    Returns an optimizer instance

    :param model: the model to optimize
    :param optimizer: one of 'SGD' or 'Adam'
    :param learning_rate: the learning rate
    :param momentum: the momentum (if the optimizer uses it)
    :param weight_decay: regularization coefficient
    """
    if optimizer.lower() == "sgd":
        # YOUR CODE HERE: create an instance of the SGD
        # optimizer. Use the input parameters learning_rate, momentum
        # and weight_decay
        opt = torch.optim.SGD(
                model.parameters(),
                lr = learning_rate,
                momentum =momentum,
                weight_decay = weight_decay
        )

    elif optimizer.lower() == "adam":
        # YOUR CODE HERE: create an instance of the Adam
        # optimizer. Use the input parameters learning_rate, momentum
        # and weight_decay
        opt = torch.optim.Adam(
                    model.parameters(), 
                    lr = learning_rate,
                    weight_decay = weight_decay
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt



def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "plateau",
    epochs: int = 30,
    eta_min: float = 1e-4,
    factor: float = 0.1,
    patience: int = 5,
    threshold: float = 0.01,
    verbose: bool = True
):
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=eta_min
        )
    elif scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            threshold=threshold,
            verbose=verbose
        )
    else:
         raise ValueError(f"Scheduler {scheduler_type} not supported")



######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def fake_model():
    return nn.Linear(16, 256)


def test_get_loss():

    loss = get_loss()

    assert isinstance(
        loss, nn.CrossEntropyLoss
    ), f"Expected cross entropy loss, found {type(loss)}"


def test_get_optimizer_type(fake_model):

    opt = get_optimizer(fake_model)

    assert isinstance(opt, torch.optim.SGD), f"Expected SGD optimizer, got {type(opt)}"


def test_get_optimizer_is_linked_with_model(fake_model):

    opt = get_optimizer(fake_model)

    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])


def test_get_optimizer_returns_adam(fake_model):

    opt = get_optimizer(fake_model, optimizer="adam")

    assert opt.param_groups[0]["params"][0].shape == torch.Size([256, 16])
    assert isinstance(opt, torch.optim.Adam), f"Expected SGD optimizer, got {type(opt)}"


def test_get_optimizer_sets_learning_rate(fake_model):

    opt = get_optimizer(fake_model, optimizer="adam", learning_rate=0.123)

    assert (
        opt.param_groups[0]["lr"] == 0.123
    ), "get_optimizer is not setting the learning rate appropriately. Check your code."


def test_get_optimizer_sets_momentum(fake_model):

    opt = get_optimizer(fake_model, optimizer="SGD", momentum=0.123)

    assert (
        opt.param_groups[0]["momentum"] == 0.123
    ), "get_optimizer is not setting the momentum appropriately. Check your code."


def test_get_optimizer_sets_weight_decat(fake_model):

    opt = get_optimizer(fake_model, optimizer="SGD", weight_decay=0.123)

    assert (
        opt.param_groups[0]["weight_decay"] == 0.123
    ), "get_optimizer is not setting the weight_decay appropriately. Check your code."
    
    
#test scheduler requires optimizer object, resulted by get_optimizer
def test_get_scheduler_cosine_scheduler_with_sgd(fake_model):
    optimizer = get_optimizer(fake_model, optimizer="SGD", learning_rate=0.02, momentum=0.9, weight_decay=1e-4)
    scheduler = get_scheduler(optimizer, scheduler_type="cosine", epochs=5, eta_min=1e-5)

    assert isinstance(optimizer, torch.optim.SGD)
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert optimizer.param_groups[0]["lr"] == 0.02

def test_get_scheduler_plateau_scheduler_with_adam(fake_model):
    optimizer = get_optimizer(fake_model, optimizer="Adam", learning_rate=0.001, weight_decay=1e-4)
    scheduler = get_scheduler(optimizer, scheduler_type="plateau", factor=0.1, patience=2, threshold=0.01)

    assert isinstance(optimizer, torch.optim.Adam)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert optimizer.param_groups[0]["lr"] == 0.001


def test_get_scheduler_scheduler_invalid_type(fake_model):
    optimizer = get_optimizer(fake_model, optimizer="SGD", learning_rate=0.01)

    with pytest.raises(ValueError) as exc_info:
        get_scheduler(optimizer, scheduler_type="invalid_scheduler")

    assert "not supported" in str(exc_info.value)

   

    



