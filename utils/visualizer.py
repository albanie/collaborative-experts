"""
TODO(samuel): Add attribution to cycle-gan codebase
"""
import os
from pathlib import Path
from . import util, html


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
        self.web_dir = log_dir
        self.img_dir = os.path.join(self.web_dir, "images")
        print(f"create web directory {self.web_dir}...")
        util.mkdirs([self.web_dir, self.img_dir])
        src_dir = Path(src_video_dir).absolute()
        print(f"symlinking videos from {src_dir}...")
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
        if not Path(self.web_dir).exists():
            Path(self.web_dir).mkdir(exists_ok=True, parents=True)
        print(f"updating webpage at {self.web_dir}")
        title = f"Experiment name = {self.name}"
        refresh = True
        if not refresh:
            print("DISABLING WEB PAGE REFRESH")
        webpage = html.HTML(web_dir=self.web_dir, title=title, refresh=refresh)

        msg = f"epoch [{epoch}] - {self.name}"
        webpage.add_header(msg)
        msg = (f"R1: {100 * metrics['R1']:.1f}, "
               f"R5: {100 * metrics['R5']:.1f}, "
               f"R10: {100 * metrics['R10']:.1f}, "
               f"MedR: {metrics['MedR']}")
        webpage.add_header(msg)
        print(f"Top {len(rankings[0])} retreived videos at epoch: {epoch}")

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
                txt = (f"{gt_captions}<br><b>Rank: {ranking['gt-rank']}, "
                       f"Sim: {ranking['gt-sim']:.3f} [{ranking['gt-path'].stem}]")
                txts.append(txt)
                links.append(gt_vid_path)
                vids.append(gt_vid_path)

            for idx, (path, sim) in enumerate(zip(ranking["top-k-paths"],
                                                  ranking["top-k-sims"])):
                if ranking["hide-gt"]:
                    txt = f"choice: {idx}"
                else:
                    txt = f"<b>Rank: {idx}, Sim: {sim:.3f}, [{path.stem}]"
                txts.append(txt)
                vid_path = str(Path("videos") / path.name)
                vids.append(vid_path)
                links.append(vid_path)
            webpage.add_videos(vids, txts, links, width=200)
        print(f"added {len(vids)} videos")
        webpage.save()
