from ..fast_movie import FastMovie


def error_catcher(
    export_frames,
    frame_export_channel,
    export_channels,
    image_range,
    frame_export_images,
):
    """
    Control function to check weather the setting are in the right
    form to export a single frame.

    Args:
        export_frames: boolean weather frames should be exported as
            images
        frame_export_channel: frame mode (e.g. 'ui' or 'df')
        export_channels: movie mode (e.g. 'udi')
        image_range: movie export range
        frame_export_images: frame index

    Returns:
        frame_export_channel: adjusted frame_export_channel

    """

    if export_frames:
        if "ud" in frame_export_channel:
            frame_export_channel = frame_export_channel[-2:]
            print(
                "Warning: frame_export_channel should not be 'ud<>'. Using {} instead.".format(
                    frame_export_channel
                )
            )
        for c in frame_export_channel:
            if c not in export_channels:
                frame_export_channel = export_channels[-2:]
                print(
                    "Warning: frame_export_channel is not a subset of export_channels.\n",
                    "frame_export_channel automatically set to {}.".format(
                        frame_export_channel
                    ),
                )
                break

    if image_range is not None:
        if export_frames:
            if type(frame_export_images) is tuple:
                if (
                    frame_export_images[0] < image_range[0]
                    or frame_export_images[1] > image_range[1]
                ):
                    print(
                        "Warning: At least one image in frame_export_images is not in 'image_range'. Some exported ",
                        "frames will be raw data!",
                    )
            else:
                if (
                    frame_export_images < image_range[0]
                    or frame_export_images > image_range[1]
                ):
                    print(
                        "Warning: frame_export_images is not in 'image_range'. Exported frames will be raw data!"
                    )

    return frame_export_channel
