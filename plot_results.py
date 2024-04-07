import matplotlib.pyplot as plt
import numpy as np
import argparse
plt.style.use("seaborn-whitegrid")

def plot_results(log_file):
    with open(log_file, "r") as f:
        lines = f.readlines()
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    for line in lines:
        if "Epoch" in line:
            line = line.strip().split(" - ")
            for l in line:
                if "Train Loss" in l:
                    train_loss.append(float(l.split(": ")[1]))
                elif "Train Acc" in l:
                    train_acc.append(float(l.split(": ")[1]))
                elif "Val Loss" in l:
                    val_loss.append(float(l.split(": ")[1]))
                elif "Val Acc" in l:
                    val_acc.append(float(l.split(": ")[1]))
                elif "Epoch" in l:
                    epoch = int(l.split(": ")[1])
                    
    print(f"train loss: {train_loss}")
    print(f"train acc: {train_acc}")
    print(f"val loss: {val_loss}")
    print(f"val acc: {val_acc}")
    
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    fig.set_tight_layout(True)
    fig.suptitle("Training and Validation Results", fontsize=16)
    ax[0].plot(np.arange(1, epoch+2), train_loss, label="train loss")
    ax[0].plot(np.arange(1, epoch+2), val_loss, label="val loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[1].plot(np.arange(1, epoch+2), train_acc, label="train acc")
    ax[1].plot(np.arange(1, epoch+2), val_acc, label="val acc")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, help="Path to log file")
    args = parser.parse_args()
    plot_results(args.log_file)