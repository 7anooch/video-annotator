Clone the project, then simply run `python annotator.py` to launch a dialog box and GUI.

A CSV with annotations (with the same name as the video file) will be written in the same directory as the video file.

Annotation does not have to be done in a single sitting, reading in the same video file will load it's corresponding .csv file so annotation can be continued/improved.

Running the script `plot.py` will launch a dialog box to select an annotatation csv file. Multiple files can be selected at once. Once the files are chosen it will plot an ethogram of the annotation(s), on a multipanel figure is several files are chosen.

I've set some keybindings for ease of use: left/right arrows go to previous/next frames. And 's', 'r', and 't' correspond to 'stop', 'run' and 'turn' labels, respectively.
