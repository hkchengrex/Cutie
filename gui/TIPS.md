### Tips

Core mechanism: annotate objects at one or more frames and use propagation to complete the video.
Use permanent memory to store accurate segmentation (commit good frames to it) for best results.
The first frame to enter the memory bank is always committed to the permanent memory.
Reset memory if needed.

- Use left-click for foreground annotation and right-click for background annotation.
- Use middle-click to toggle visualization target (for layered, popout, and binary mask export).
- Use number keys or the spinbox to change the object to be operated on. If it does not respond, most likely the correct number of objects was not specified during program startup.
- "Export as video" only aggregates visualizations that are saved on disks. You need to check "save overlay" for that to happen.
- Exported binary/soft masks can be used in other applications like ProPainter. Note inpainting prefer over-segmentation over under-segmentation -- use a larger dilation radius if needed
- Memory can be corrupted by bad segmentations. Make good use of "reset memory" and do not commit bad segmentations.
- The "layered" visualization mode inserts an RGBA layer between the foreground and the background. Use "import layer" to select a new layer.

Issues and further documentation: hkchengrex/Cutie
