import click
from .commands.gradcam import gradcam_cmd

@click.group()
@click.pass_context
def cli(ctx):
    """
    Can be used to set global config options for all commands.
    """

    ctx.ensure_object(dict)

@cli.command()
@click.option("-i", "--input_mp4", required=True, help="The path to an MP4 input file.")
@click.option("-o", "--output_mp4", required=True, help="The path to an MP4 output file.")
@click.option("-m", "--model", required=True, help="The path to a model.pb file.")
@click.option("-mm", "--model_metadata", required=True, help="The path to a model metadata file.")
@click.option("--fps", type=int, default=15, help="The path to a model metadata file.")
@click.pass_context
def gradcam(ctx, input_mp4, output_mp4, model, model_metadata, fps):
    """
    Process an input MP4 by applying gradcam to each frame of the video using the provided model.
    Outputs a video with a gradcam overlay.
    """

    gradcam_cmd(
        input_file=input_mp4, 
        output_file=output_mp4,
        model=model, 
        model_metadata=model_metadata, 
        fps=fps
    )