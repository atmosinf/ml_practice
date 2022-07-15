import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YOLOLoss

# torch.backend.cudnn.benchmark = True

def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []

    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2, = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE)
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[0], y0, scaled_anchors[1])
                + loss_fn(out[0], y0, scaled_anchors[2])
            )
            losses.append(loss.item())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update progress bar
            mean_loss = sum(losses) / len(losses)
            loop.set_postfix(loss = mean_loss)


def main():
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr = config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    train_loader, test_loader, train_eval_loader = get_loaders(
        train_csv_path = '../YOLOv1/archive (2)/100examples.csv', test_csv_path = '../YOLOv1/archive (2)/100examples.csv'
    )
    loss_fn = YOLOLoss()
    scaler = torch.cuda.amp.GradScaler()

    # if config.LOAD_MODEL:
        # load_checkpoint(config.CHECKPOINT_FILE, model, optimizer, config.LEARNING_RATE)

    scaled_anchors = (
        torch.tensor(config.ANCHORS)
        * torch.tensor(config.S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)
    ).to(config.DEVICE)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(test_loader, model, optimizer, loss_fn, scaler, scaled_anchors) # train on the test_loader, because it is easier and we're not actually training the network

        # if config.SAVE_MODEL:
            # save_checkpoint(model, optimizer)


if __name__ == '__main__':
    main()