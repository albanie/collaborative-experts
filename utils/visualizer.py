"""
TODO(samuel): Add attribution to cycle-gan codebase
"""
import os
import sys
import textwrap
import ntpath
import time
from pathlib import Path
from . import util, html
from subprocess import Popen, PIPE
import numpy as np
from scipy.misc import imresize

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data)
        image_name = "%s_%s.png" % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, _ = im.shape
        if aspect_ratio > 1.0:
            im = imresize(im, (h, int(w * aspect_ratio)), interp="bicubic")
        if aspect_ratio < 1.0:
            im = imresize(im, (int(h / aspect_ratio), w), interp="bicubic")
        util.save_image(im, save_path)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer:
    """This class includes several functions that can display/save images.

    It uses a Python library 'visdom' for display, and a Python library 'dominate'
    (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, exp_name, log_dir, src_video_dir):
        """Initialize the Visualizer class
        Create an HTML object for saveing HTML filters
        """
        self.name = exp_name
        # create an HTML object at <log_dir>/web/; images will be saved
        # under <log_dir>/web/images/
        self.web_dir = log_dir
        self.img_dir = os.path.join(self.web_dir, "images")
        print("create web directory %s..." % self.web_dir)
        util.mkdirs([self.web_dir, self.img_dir])
        src_dir = Path(src_video_dir).absolute()
        print("symlinking videos from {}...".format(src_dir))
        sym_dir = (Path(self.web_dir) / "videos").absolute()
        if sym_dir.is_symlink():
            os.remove(sym_dir)
        sym_dir.symlink_to(src_dir)

    def display_current_results(self, rankings, epoch, metrics):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        # save images to the disk
        # for label, image in visuals.items():
        #     image_numpy = util.tensor2im(image)
        #     img_path = os.path.join(
        #         self.img_dir, "epoch%.3d_%s.png" % (epoch, label)
        #     )
        #     util.save_image(image_numpy, img_path)

        if not Path(self.web_dir).exists():
            Path(self.web_dir).mkdir(exists_ok=True, parents=True)
        print("updating webpage at {}".format(self.web_dir))
        title = "Experiment name = {}".format(self.name)
        refresh = True
        if not refresh:
            print("DISABLING WEB PAGE REFRESH")
        webpage = html.HTML(web_dir=self.web_dir, title=title, refresh=refresh)

        # for n in range(epoch, 0, -1):
        msg = "epoch [{}] - {}"
        msg = msg.format(epoch, self.name)
        webpage.add_header(msg)
        msg = "R1: {:.1f}, R5: {:.1f}, R10: {:.1f}, MedR: {}"
        msg = msg.format(100 * metrics["R1"],
                         100 * metrics["R5"],
                         100 * metrics["R10"],
                         metrics["MedR"])
        webpage.add_header(msg)
        print("Top {} retreived videos at epoch: {}".format(len(rankings[0]), epoch))

        for ranking in rankings:
            vids, txts, links = [], [], []
            gt_vid_path = str(Path("videos") / ranking["gt-path"].name)
            gt_captions = [" ".join(x) for x in ranking["gt-captions"]]
            gt_captions = "<br>".join(gt_captions)
            if ranking["hide-gt"]:
                txts.append(gt_captions)
                links.append("hidden")
                vids.append("hidden")
            else:
                txt = "{}<br><b>Rank: {}, Sim: {:.3f} [{}]"
                txt = txt.format(gt_captions, ranking["gt-rank"], ranking["gt-sim"],
                                 ranking["gt-path"].stem)
                txts.append(txt)
                links.append(gt_vid_path)
                vids.append(gt_vid_path)

            for idx, (path, sim) in enumerate(zip(ranking["top-k-paths"],
                                                  ranking["top-k-sims"])):
                if ranking["hide-gt"]:
                    txt = "choice: {}".format(idx)
                else:
                    txt = "<b>Rank: {}, Sim: {:.3f}, [{}]"
                    txt = txt.format(idx, sim, path.stem)
                txts.append(txt)
                vid_path = str(Path("videos") / path.name)
                vids.append(vid_path)
                links.append(vid_path)
            webpage.add_videos(vids, txts, links, width=200)
        print("added {} videos".format(len(vids)))
        webpage.save()
