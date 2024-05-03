### Tips

Core mechanism: annotate objects at one or more frames and use propagation to complete the video.
Use permanent memory to store accurate segmentation (commit good frames to it) for best results.
The first frame to enter the memory bank is always committed to the permanent memory.
Reset memory if needed.

Controls:

- Use left-click for foreground annotation and right-click for background annotation.
- Use number keys or the spinbox to change the object to be operated on. If it does not respond, most likely the correct number of objects was not specified during program startup.
- Use left/right arrows to move between frames, shift+arrow to move by 10 frames, and alt/option+arrow to move to the start/end.
- Use F/space and B to propagate forward and backward, respectively.
- Use C to commit a frame to permanent memory.
- Memory can be corrupted by bad segmentations. Make good use of "reset memory" and do not commit bad segmentations.
- "Export as video" only aggregates visualizations that are saved on disks. You need to check "save overlay" for that to happen.

Visualizations:

- Middle-click on target objects to toggle some visualization effects (for layered, popout, RGBA, and binary mask export).
- Soft masks are only saved for the "propagated" frames, not for the interacted frames. To save all frames, utilize forward and backward propagation.
- For some visualizations (layered and RGBA), the images saved during propagation will be higher quality with soft edges. This is because we have access to the soft mask only during propagation. Set the save visualization mode to "Propagation only" to only save during propagation.
- The "layered" visualization mode inserts an RGBA layer between the foreground and the background. Use "import layer" to select a new layer.

Exporting:

- Exported binary/soft masks can be used in other applications like ProPainter. Note inpainting prefer over-segmentation over under-segmentation -- use a larger dilation radius if needed

Issues and further documentation: hkchengrex/Cutie
