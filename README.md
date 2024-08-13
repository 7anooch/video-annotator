Clone the project, then simply run `python annotator.py` to launch a dialog box and GUI.

I think this only supports .avi videos currently.

A CSV with annotations (with the same name as the video file) will be written in the same directory as the video file.

Annotation does not have to be done in a single sitting, reading in the same video file will load it's corresponding .csv file so annotation can be continued/improved.

I've set some keybindings for ease of use: left/right arrows go to previous/next frames. And 's', 'r', and 't' correspond to 'stop', 'run' and 'turn' labels, respectively.
