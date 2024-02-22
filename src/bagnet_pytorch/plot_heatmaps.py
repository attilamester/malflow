import bagnet_pytorch.pytorchnet as pytorchnet
import cv2
import database.dataloader as dl
import matplotlib.pyplot as plt
import numpy as np
import torch
from database.config import dataset_config as config
from skimage import feature, transform


def plot_heatmap(
    heatmap,
    original,
    ax,
    cmap="RdBu_r",
    percentile=99,
    dilation=0.5,
    alpha=0.25,
):
    """
    Plots the heatmap on top of the original image
    (which is shown by most important edges).

    Parameters
    ----------
    heatmap : Numpy Array of shape [X, X]
        Heatmap to visualise.
    original : Numpy array of shape [X, X, 3]
        Original image for which the heatmap was computed.
    ax : Matplotlib axis
        Axis onto which the heatmap should be plotted.
    cmap : Matplotlib color map
        Color map for the visualisation of the heatmaps (default: RdBu_r)
    percentile : float between 0 and 100 (default: 99)
        Extreme values outside of the percentile range are clipped.
        This avoids that a single outlier dominates the whole heatmap.
    dilation : float
        Resizing of the original image. Influences the edge detector and
        thus the image overlay.
    alpha : float in [0, 1]
        Opacity of the overlay image.

    """
    if len(heatmap.shape) == 3:
        heatmap = np.mean(heatmap, 0)

    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, heatmap.shape[1], dx)
    yy = np.arange(0.0, heatmap.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_original = plt.get_cmap("Greys_r")
    cmap_original.set_bad(alpha=0)
    overlay = None
    if original is not None:
        # Compute edges (to overlay to heatmaps later)
        original_greyscale = (
            original
            if len(original.shape) == 2
            else np.mean(original, axis=-1)
        )
        in_image_upscaled = transform.rescale(
            original_greyscale,
            dilation,
            mode="constant",
            # multichannel=False, # no longer supported!
            anti_aliasing=True,
        )
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(heatmap), percentile)
    abs_min = abs_max

    ax.imshow(
        heatmap,
        extent=extent,
        interpolation="none",
        cmap=cmap,
        vmin=-abs_min,
        vmax=abs_max,
    )
    if overlay is not None:
        ax.imshow(
            overlay,
            extent=extent,
            interpolation="none",
            cmap=cmap_original,
            alpha=alpha,
        )


def plot_heatmap_protopnet_style(
    heatmap, original, ax, filename, pred, true, actual, blur=False, perc=99
):
    if len(heatmap.shape) == 3:
        heatmap = np.mean(heatmap, 0)

    # cutting off large pixel values
    perc = np.percentile(abs(heatmap), perc)
    heatmap[heatmap > perc] = perc
    heatmap[heatmap < -perc] = -perc

    vmin = np.amin(heatmap)
    vmax = np.amax(heatmap)
    heatmap = (heatmap - vmin) / (vmax - vmin)

    heatmap = np.exp(7 * heatmap)  # scaling pixel intensities

    vmin = np.amin(heatmap)
    vmax = np.amax(heatmap)
    heatmap = (heatmap - vmin) / (vmax - vmin)

    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    if blur:
        heatmap = cv2.blur(heatmap, (2, 2))
    heatmap = np.float32(heatmap) / 255
    # heatmap = heatmap[
    #     ..., ::-1
    # ]  # inverting last dimension??? - inverts colors
    overlayed_img = 0.5 * original + 0.3 * heatmap

    cv2.imwrite
    (
        f"./heatmaps/pascalvoc-inv/{filename}_{pred}_{true}_{actual}.png",
        overlayed_img * 255,
    )

    ax.imshow(overlayed_img)


def generate_heatmap_pytorch(
    model, image, target, patchsize, padding="replication"
):
    """
    Generates high-resolution heatmap for a BagNet by decomposing the
    image into all possible patches and by computing the logits for
    each patch.

    Parameters
    ----------
    model : Pytorch Model
        This should be one of the BagNets.
    image : Numpy array of shape [1, 3, X, X]
        The image for which we want to compute the heatmap.
    target : int
        Class for which the heatmap is computed.
    patchsize : int
        The size of the receptive field of the given BagNet.

    """
    import torch

    with torch.no_grad():
        _, c, x, y = image.shape
        print("Image size:")
        print(c, x, y)

        if padding == "zero":  # original code
            padded_image = np.zeros((c, x + patchsize - 1, y + patchsize - 1))
            padded_image[
                :,
                (patchsize - 1) // 2 : (patchsize - 1) // 2 + x,
                (patchsize - 1) // 2 : (patchsize - 1) // 2 + y,
            ] = image[0]
            input = torch.from_numpy(padded_image[None].astype(np.float32))
        elif padding == "replication":
            padder = torch.nn.ReplicationPad2d((patchsize - 1) // 2)
            input = padder(torch.from_numpy(image))
        else:
            input = torch.from_numpy(image)

        # extract patches
        patches = input.permute(0, 2, 3, 1)
        patches = patches.unfold(1, patchsize, 1).unfold(2, patchsize, 1)
        patches = patches.contiguous().view(
            (-1, config["color_channels"], patchsize, patchsize)
        )

        # compute logits for each patch
        logits_list = []

        for batch_patches in torch.split(patches, 10000):
            print(f"Batch: {batch_patches.shape}")
            logits = model(batch_patches.to(next(model.parameters()).device))
            logits = logits[:, target]
            logits_list.append(logits.data.cpu().numpy().copy())

        # logits = np.hstack(logits_list)
        logits = np.vstack(logits_list)

        delta = (patchsize - 1) if padding is None else 0

        return logits.reshape(
            (config["img_shape"][0] - delta, config["img_shape"][1] - delta)
        )


batch_size = 1  # todo: read this from config
patch_size = 17

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = pytorchnet.bagnet17(
        pretrained=True, num_classes=config["num_classes"]
    )

    # loc = 'cuda:{}'.format(0)

    # checkpoint = torch.load(
    #   "/home/miafranc/bosch/bagnet/model_best_bagnet17-bosch.pth.tar",
    #   map_location=device)
    checkpoint = torch.load(
        "./models/checkpoint_bagnet_pascalvoc_lre4.pth.tar",
        map_location=device,
    )

    model.load_state_dict(checkpoint["state_dict"])

    print("checkpoint stats:")
    print(checkpoint["epoch"])
    print(checkpoint["best_acc1"])

    # print("The model will be running on", device, "device")
    # # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    model.eval()

    val_loader = dl.test_loader

    filenumber = 0

    # for i, (images, target, filename) in enumerate(val_loader):
    for i, (images, target) in enumerate(val_loader):
        # filename = filename[0].replace("/", "__")
        # filename = filename.replace(".png", "")

        print(i)
        image = images.numpy()

        # print(image.shape)
        #
        # image = np.reshape(np.mean(image, axis=1), (1, 3, 224, 224))
        # image = np.reshape(
        #     image[:, 0, :, :],
        #     (
        #         batch_size,
        #         config["color_channels"],
        #         config["img_shape"][0],
        #         config["img_shape"][1],
        #     ),
        # )
        # print("Target:")
        # print(target)
        target_num = target.detach().cpu().numpy()[0]

        images = images.to(device, non_blocking=True)
        output = model(images)
        # print("Predicted:")
        pr = output.detach().cpu().numpy()[0].argmax()
        pred = np.append([], int(pr))
        # print(pred)
        # print(torch.from_numpy(np.array(pred)))

        # heatmap_pred = generate_heatmap_pytorch(model,
        #   image,
        #   pred,
        #   patch_size,
        #   padding='replication')
        # heatmap_target = generate_heatmap_pytorch(model,
        #   image,
        #   target,
        #   patch_size,
        #   padding='replication')

        heatmap_0 = generate_heatmap_pytorch(
            model, image, [target_num], patch_size, padding="replication"
        )
        heatmap_1 = generate_heatmap_pytorch(
            model, image, [pr], patch_size, padding="replication"
        )
        heatmap_2 = generate_heatmap_pytorch(
            model, image, [pr], patch_size, padding="replication"
        )

        # plot heatmap
        fig = plt.figure(figsize=(8, 4))

        original_image = image[0].transpose([1, 2, 0])

        ax = plt.subplot(311)
        # ax.set_title('original')
        # plt.imshow(original_image / 255.)
        # plt.imshow(original_image, cmap='Greys_r')
        plot_heatmap_protopnet_style(
            heatmap_0,
            original_image,
            ax,
            filenumber,
            target_num,
            pr,
            pr,
            80,
            False,
        )
        plt.axis("off")

        ax = plt.subplot(312)
        # ax.set_title(pr, fontsize=12)
        # plot_heatmap(
        #     heatmap, original_image, ax, dilation=0.5, percentile=99, alpha=0.25
        #     # heatmap, None, ax, dilation=0.5, percentile=99, alpha=0.25
        # )
        # plot_heatmap_protopnet_style(heatmap_pred, original_image, ax, 98, False)
        plot_heatmap_protopnet_style(
            heatmap_1,
            original_image,
            ax,
            filenumber,
            target_num,
            pr,
            target_num,
            80,
            False,
        )
        plt.axis("off")

        ax = plt.subplot(313)
        # ax.set_title(target.detach().cpu().numpy()[0], fontsize=12)
        # plot_heatmap(
        #     heatmap, original_image, ax, dilation=0.5, percentile=99, alpha=0.25
        #     # heatmap, None, ax, dilation=0.5, percentile=99, alpha=0.25
        # )
        # plot_heatmap_protopnet_style(heatmap_target, original_image, ax, 98, False)
        plot_heatmap_protopnet_style(
            heatmap_2,
            original_image,
            ax,
            filenumber,
            target_num,
            target_num,
            target_num,
            80,
            False,
        )
        plt.axis("off")

        # plt.show()
        # plt.savefig(f'{i}.png')
        plt.savefig(
            f"./heatmaps/pascalvoc-inv/{filenumber}_plot_{target_num}_{pr}.png"
        )

        filenumber = filenumber + 1
