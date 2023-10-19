# Interactive Tool

Start the interactive tool with `python interactive_demo.py`.

It takes the following arguments:

- `images`: Path to a directory containing images
- `video`: Path to a video file
- `workspace`: Path to a directory for saving the results
- `num_objects`: Number of objects to segment

We obtain the frames using the following rules:

1. Priority 1: If a "images" folder exists in the workspace, we will read from that directory
2. Priority 2: If --images is specified, we will copy/resize those images to the workspace
3. Priority 3: If --video is specified, we will extract the frames to the workspace (in an "images" folder) and read from there

In any case, if a "masks" folder exists in the workspace, we will use that to initialize the mask.
That way, you can continue annotation from an interrupted run as long as the same workspace is used.

There are additional configurations that you can modify in `cutie/config/gui_config.yaml`. You cannot use command line for overriding those. The default should work most of the times.

See [TIPS](../gui/TIPS.md) for some tips on using the tool.
