import re

from matplotlib import pyplot as plt


def parse_log(file_path):
    """
    Parse the log file to extract training metrics.

    Parameters
    ----------
    file_path : str
        The path to the log file.

    Returns
    -------
    tuple
        A tuple containing lists of iterations, dc_losses, nc_losses, rec_losses,
        rec_losses_aug, total_losses, and accuracies.
    """
    iterations = []
    dc_losses = []
    nc_losses = []
    rec_losses = []
    rec_losses_aug = []
    total_losses = []
    accuracies = []

    with open(file_path, "r") as file:
        for line in file:
            if "moving averages" in line:
                iteration = int(re.search(r"(\d+) - moving averages", line).group(1))
                dc_loss = float(re.search(r"dc_loss: ([\d.]+)", line).group(1))
                nc_loss = float(re.search(r"nc_loss: ([\d.]+)", line).group(1))
                rec_loss = float(re.search(r"rec_loss: ([\d.]+)", line).group(1))
                rec_loss_aug_match = re.search(r"rec_loss_aug: ([\d.]+)", line)
                if rec_loss_aug_match:
                    rec_loss_aug = float(rec_loss_aug_match.group(1))
                else:
                    rec_loss_aug = None
                total_loss = float(re.search(r"total_loss: ([\d.]+)", line).group(1))
                accuracy_match = re.search(r"accuracy: ([\d.]+)", line)
                if accuracy_match:
                    accuracy = float(accuracy_match.group(1))
                else:
                    accuracy = None

                iterations.append(iteration)
                dc_losses.append(dc_loss)
                nc_losses.append(nc_loss)
                rec_losses.append(rec_loss)
                rec_losses_aug.append(rec_loss_aug)
                total_losses.append(total_loss)
                accuracies.append(accuracy)

    return (
        iterations,
        dc_losses,
        nc_losses,
        rec_losses,
        rec_losses_aug,
        total_losses,
        accuracies,
    )


def plot_metrics(log_file_path):
    """
    Plot training metrics from a log file.

    Parameters
    ----------
    log_file_path : str
        The path to the log file.
    """
    (
        iterations,
        dc_losses,
        nc_losses,
        rec_losses,
        rec_losses_aug,
        total_losses,
        accuracies,
    ) = parse_log(log_file_path)

    plt.figure(figsize=(12, 6))
    title = log_file_path.split("/")[-1].replace(".txt", "")
    plt.suptitle(title)

    plt.subplot(2, 1, 1)
    plt.plot(iterations, dc_losses, label="DC Loss")
    plt.plot(iterations, nc_losses, label="NC Loss")
    plt.plot(iterations, rec_losses, label="Reconstruction Loss")
    plt.plot(iterations, rec_losses_aug, label="Reconstruction Loss Augmented")
    plt.plot(iterations, total_losses, label="Total Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(iterations, accuracies, label="Accuracy", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_comparison(log_file_path1, log_file_path2):
    """
    Plot comparison of training metrics from two log files.

    Parameters
    ----------
    log_file_path1 : str
        The path to the first log file.
    log_file_path2 : str
        The path to the second log file.
    """
    (
        iterations1,
        dc_losses1,
        nc_losses1,
        rec_losses1,
        rec_losses_aug1,
        total_losses1,
        accuracies1,
    ) = parse_log(log_file_path1)
    (
        iterations2,
        dc_losses2,
        nc_losses2,
        rec_losses2,
        rec_losses_aug2,
        total_losses2,
        accuracies2,
    ) = parse_log(log_file_path2)

    plt.figure(figsize=(12, 15))

    title1 = log_file_path1.split("/")[-1].replace(".txt", "")
    title2 = log_file_path2.split("/")[-1].replace(".txt", "")

    plt.subplot(5, 2, 1)
    plt.plot(iterations1, dc_losses1, label=f"{title1} DC Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title1} DC Loss")
    plt.legend()

    plt.subplot(5, 2, 2)
    plt.plot(iterations2, dc_losses2, label=f"{title2} DC Loss", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title2} DC Loss")
    plt.legend()

    plt.subplot(5, 2, 3)
    plt.plot(iterations1, nc_losses1, label=f"{title1} NC Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title1} NC Loss")
    plt.legend()

    plt.subplot(5, 2, 4)
    plt.plot(iterations2, nc_losses2, label=f"{title2} NC Loss", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title2} NC Loss")
    plt.legend()

    plt.subplot(5, 2, 5)
    plt.plot(iterations1, rec_losses1, label=f"{title1} Reconstruction Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title1} Reconstruction Loss")
    plt.legend()

    plt.subplot(5, 2, 6)
    plt.plot(
        iterations2, rec_losses2, label=f"{title2} Reconstruction Loss", color="orange"
    )
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title2} Reconstruction Loss")
    plt.legend()

    plt.subplot(5, 2, 7)
    plt.plot(
        iterations1, rec_losses_aug1, label=f"{title1} Reconstruction Loss Augmented"
    )
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title1} Reconstruction Loss Augmented")
    plt.legend()

    plt.subplot(5, 2, 8)
    plt.plot(
        iterations2,
        rec_losses_aug2,
        label=f"{title2} Reconstruction Loss Augmented",
        color="orange",
    )
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title2} Reconstruction Loss Augmented")
    plt.legend()

    plt.subplot(5, 2, 9)
    plt.plot(iterations1, total_losses1, label=f"{title1} Total Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title1} Total Loss")
    plt.legend()

    plt.subplot(5, 2, 10)
    plt.plot(iterations2, total_losses2, label=f"{title2} Total Loss", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title2} Total Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_accuracy_comparison(log_file_path1, log_file_path2):
    """
    Plot comparison of training accuracy from two log files.

    Parameters
    ----------
    log_file_path1 : str
        The path to the first log file.
    log_file_path2 : str
        The path to the second log file.
    """
    iterations1, _, _, _, _, _, accuracies1 = parse_log(log_file_path1)
    iterations2, _, _, _, _, _, accuracies2 = parse_log(log_file_path2)

    plt.figure(figsize=(12, 6))

    title1 = log_file_path1.split("/")[-1].replace(".txt", "")
    title2 = log_file_path2.split("/")[-1].replace(".txt", "")

    plt.plot(iterations1, accuracies1, label=f"{title1} Accuracy")
    plt.plot(iterations2, accuracies2, label=f"{title2} Accuracy", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Comparison")
    plt.legend()

    plt.show()
